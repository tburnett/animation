"""Fermi weekly flux or exposure maps 

Generate a dict indexed by week number of nside 128 pixels

weeks start from 1, index 0 has the total.

To run this:
```
    python -m animation.map_job <jobname>
```
where <jobname> must contain either 'flux' or 'exposure'

The file will be written to <filepath>/weekly_<jobname>_maps.pkl.
"""
import os,sys
from pathlib import Path
from time import localtime, strftime
import numpy as np
try:
    from  simple_slurm import Slurm
except:
    Slurm=None

filepath = os.environ.get('ANIMATION_FILES', 'files')
filepath = Path(filepath).expanduser()
assert filepath.is_dir(), 'Expect either $ANIMATION_FILES  or files to be folder'

def check_jobname(jobname):
    for name in 'exposure flux'.split():
        if jobname.find(name)>=0: return True
    return False

def run_job(jobname):
    """Create the dict, using the Slurm batch system if available
    """
    from wtlike import DataView, WeightedAeff, FermiInterval, Timer
    import pickle

    et = Timer()
    if jobname.find('flux')>=0:
        mapper = lambda dt: DataView(dt).flux_map(beam_window=WeightedAeff().beam_window() ).astype(np.float32)
    elif jobname.find('exposure')>=0:
        mapper = lambda dt: DataView(dt).exposure_map(beam_window=WeightedAeff().beam_window() ).astype(np.float32)
    else:
        raise Exception(f'Bad jobname: {jobname}: must include either "flux" or "exposure"')
    delta_t=7
    print(f'Start job {jobname} at {strftime("%Y-%m-%d %H:%M:%S", localtime())}')

    map_dict={}
    # 0 is the full map
    map_dict[0] = mapper((0,0))

    for k, time_range in enumerate( FermiInterval(delta_t)):
        try:
            map_dict[k+1] = mapper(time_range)
        except Exception as e:
            print(f'Quitting at week {k+1}  since {e}')
            break
    
    outfile = filepath/(f'weekly_{jobname}_maps.pkl')
    with open(outfile, 'wb') as out:
        pickle.dump(map_dict, out)   

    print(et)     

def main():

    jobid = os.getenv('SLURM_JOB_ID', 0)
    name = sys.argv[0] 
    if len(sys.argv)>1:
        jobname = sys.argv[1]
    else:
        jobname = os.getenv('SLURM_JOB_NAME', None)


    print (f'{strftime("%Y-%m-%d %H:%M:%S", localtime())}')
    print(f"""name: {name}\njobid: {jobid}""")

    if jobid==0:
        # submit job
        assert jobname is not None, 'Must be run interactively with a jobname'
        assert check_jobname(jobname), 'Job name must contain "flux" or "exposure"'
        module = name.replace('/', '.').replace('.py','')
        outpath = filepath/'logs'
        outpath.mkdir(exist_ok=True )

        if Slurm is not None:
            print(f'Start job {jobname} with module {module}')
            slurm = Slurm(
                job_name=jobname,
                output=outpath/f'{jobname}_%A.log',
                time='30:00',
                ntasks=1,
                )
            jobid = slurm.sbatch(f'python -m {module}')
        else:
            print(f'Running {jobname} interactively' )
            run_job(jobname)

    else:
        run_job(jobname)

if __name__=='__main__':
    main()
