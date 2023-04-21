"""
"""
import sys
import numpy as np
from pathlib import Path
from time import localtime, strftime
from wtlike import FermiInterval, Timer


class SlurmJob:
    """
    Superclass that supports Slurm Jobs
    
    The subclass must override __call__ to process an item, 0-indexed as 0-(total-1)
    This class can break up the sequence into "tasks", of `task_size` each, for submission
    (Tasks are 1-indexed)
        
    """
    def __init__(self, jobname, total, task_size=0,
                  job_time='1:00:00'
                 ):
        if task_size==0: task_size=total
        self.start = np.append(np.arange(0, total, min(task_size,total)), total)
        self.tasks = len(self.start)-1
        # for submit
        self.jobname=jobname
        self.job_time=job_time
        
    def __repr__(self):
        return f'Job: run items 0-{self.start[-1]-1} using {self.tasks} task(s) of {self.start[1]}'
        
    def __getitem__(self, k):
        # if k>= len(self.start):
        #     raise StopIteration
        return  list(range(self.start[k], self.start[k+1]))
    
    def __len__(self): 
        return self.start[-1]
    
    def __call__(self, n):
        """ demo-should be overriden"""
        print(f'\tItem {n:2d}')
     
    def task(self, task_id):
        """Run the items associated with this task
        """
        for item in self[task_id-1]:
            self(item)


    def submit(self):

        myfile = Path(sys.argv[0])
        module = myfile.parent.stem+'.'+myfile.stem
        print(f'module: {module}')
        try:
            from simple_slurm import Slurm
        except:
            print('Slurm and simple-slurm are not available', file=sys.stderr)
            if self.jobname=='test':
                print(f'Test mode for {self}')

                self.task(1)
            return 0
                  

        output=self.outpath/'logs/%A_%a.log'
        # print(f'log output to {output} ')
        print(f'Running module {module}')
        slurm = Slurm(
            job_name=self.jobname,
            output=output,
            time=self.job_time,
            array = f'1-{len(self)}',
            ntasks=len(self),
            )
        jobid =  slurm.sbatch(f'python -m {module}')
        return jobid

    @classmethod
    def main(cls, total):
        """
        Module execution, either online to submit or from Slurm to run task(s)
        """
        if len(sys.argv)==1:
            # no args
            jobname = os.getenv('SLURM_JOB_NAME', None)
            jobid = os.getenv('SLURM_JOB_ID', 0)
        else:
            # with an arg, used as jobname?
            jobname = sys.argv[1]
            jobid = 0 

        print (f'{strftime("%Y-%m-%d %H:%M:%S", localtime())}')
        print(f"""name: {jobname}\njobid: {jobid}""")

        job = cls(jobname, total)

        if jobid==0:
            # submit job(s)
            jobid  = job.submit()
            if jobid is None:
                print(f'*** Failed to submit job {jobname}')
            else:
                print(f'Submitted {jobname}:{jobid} with {len(job)} tasks')
        else:
            # called by Slurm -- get the task id.
            task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
            print(f'Start task {task_id}')
            et = Timer()
            job(int(task_id))
            print(et)
