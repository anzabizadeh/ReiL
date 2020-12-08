# -*- coding: utf-8 -*-
'''
Submit projects
===============

Submit projects to Argon


'''

from compute_resources import job_scheduler

#TODO: disable_gpu is duplicate. The other one is in `warfarin_main_iterable.py`
disable_gpu = True


def main():
    scheduler = job_scheduler.JobScheduler('sadjad-anzabizadeh@uiowa.edu')

    project_list = []
    job_list = [job_scheduler.Job(job_name=project_name,
                    priority=job_scheduler.LOW, slots=1, gpu_flag=not disable_gpu,
                    script=f'python warfarin_main_iterable.py --project_name={project_name}',
                    prereq_script='\n'.join(('module load python/3.7.0_parallel_studio-2017.4',
                        'python3 setup.py install --user'))) for project_name in project_list]

    scheduler.submit(job_list)

if __name__ == "__main__":
    main()