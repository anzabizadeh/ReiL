import dataclasses
import itertools
import os
import subprocess
from typing import Dict, List, Optional, Sequence

HIGH = 3
MEDIUM = 2
LOW = 1


@dataclasses.dataclass(order=True)
class Job:
    job_score: int = dataclasses.field(init=False, repr=False)
    job_name: str
    priority: int
    slots: int
    gpu_flag: bool = False
    queue_list: Optional[List[str]] = None
    script: Optional[str] = None
    prereq_script: Optional[str] = None

    def __post_init__(self):
        self.job_score = -(self.priority * 100 + self.slots)


class JobScheduler:
    def __init__(self,
                 email: Optional[str] = None,
                 script_template: Optional[str] = None,
                 available_queues: Optional[Sequence[str]] = None) -> None:
        # if os.name != 'posix':
        #     raise OSError('JobScheduler works only on linux!')


        if script_template is None:
            self._script_template = r'\n'.join([
                '#!/bin/bash',
                '',
                '#Set the name of the job. This will be the first part of the error/output filename.',
                '#$ -N {job.job_name}',
                '',
                '#Set the shell that should be used to run the job.',
                '#$ -S /bin/bash',
                '',
                '#Set the current working directory as the location for the error and output files.',
                '#(Will show up as .e and .o files)',
                '#$ -cwd',
                '',
                '#Select the queue to run in',
                '#$ -q {job.queue_list[0]}',
                '#$ -l gpu={job.gpu_flag}',
                '',
                '#Select the number of slots the job will use',
                '#$ -pe smp {job.slots}',
                '',
                '#Print information from the job into the output file',
                '/bin/echo Here I am: `hostname`. Starting now at: `date`',
                '',
                '#Send e-mail at beginning/end/suspension of job',
                '#$ -m bes',
                '',
                '#E-mail address to send to',
                '#$ -M {email}',
                '',
                '#code',
                '{job.prereq_script}',
                '{job.script}',
                '',
                '#Print the end date of the job before exiting',
                'echo Finished on: `date`',
            ])
        else:
            self._script_template = script_template

        if email is None and any(e in self._script_template for e in ['-M', 'email']):
            raise ValueError('script_template requires an email address. email is currently None.')

        if email is not None:
            if email.find('@') == -1 or \
                    email[email.rfind('.')+1:] not in ['com', 'edu', 'org']:
                raise ValueError('wrong email format.')

        self._email = email

        if available_queues is None:
            p = subprocess.run(['whichq'], shell=True, capture_output=True)
            self.available_queues = p.stdout.decode('ascii').split('\n')[3:]
        else:
            self.available_queues = available_queues

    def available_capacity(self,
                           queue_list: Optional[Sequence[str]] = None) -> Dict[str, int]:
        temp_list = self.available_queues if queue_list is None else queue_list
        capacities = {}
        for q in temp_list:
            try:
                p = subprocess.run(
                    f'qstat -g c -q {q}', shell=True, capture_output=True)
                capacities[q] = int(p.stdout.decode(
                    'ascii').split('\n')[2].split()[4])
            except:
                capacities[q] = 3

        return capacities

    def submit(self, job_list: List[Job], enforce_gpu_flag: bool = True) -> None:
        job_list_prioritized = self.prioritize(job_list, enforce_gpu_flag)
        scripts_list = self.generate_script_files(job_list_prioritized)
        self.run_script_files(scripts_list)
        self.remove_script_files(scripts_list)

    def prioritize(self, job_list: List[Job], enforce_gpu_flag: bool = True) -> List[Job]:
        job_list_sorted = sorted(job_list)
        for job in job_list_sorted:
            if enforce_gpu_flag and job.gpu_flag and job.queue_list is not None:
                job.queue_list = list(
                    q for q in job.queue_list if 'gpu' in q.lower())

        if any(job.queue_list is None for job in job_list_sorted):
            needed_queues = None
        else:
            needed_queues = list(
                set(itertools.chain.from_iterable(job.queue_list for job in job_list)))

        capacities = self.available_capacity(needed_queues)

        job_list_prioritized = []
        for job in job_list_sorted:
            queue_list = job.queue_list if job.queue_list is not None else self.available_queues

            for q in queue_list:
                if capacities[q] >= job.slots:
                    capacities[q] -= job.slots
                    job.queue_list = [q]
                    job_list_prioritized.append(job)
                break

        for job in job_list_sorted:
            if job not in job_list_prioritized:
                job.queue_list = job.queue_list[0] if job.queue_list is not None else self.available_queues[0]
                job_list_prioritized.append(job)

        return job_list_prioritized

    def generate_script_files(self, job_list: List[Job]) -> List[str]:
        temp_dict = dict((f'auto_generated_script_{job.job_name}.sh',
                          self._script_template.format(job=job, email=self._email))
                         for job in job_list)

        for filename, script in temp_dict.items():
            with open(filename, 'w+') as f:
                f.write(script)

        return list(temp_dict)

    def remove_script_files(self, scripts_list: List[str]) -> None:
        for script in scripts_list:
            os.remove(script)

    def run_script_files(self, scripts_list: List[str], verbose: bool = True) -> None:
        for script in scripts_list:
            p = subprocess.run(
                f'qsub {script}', shell=True, capture_output=True)
            if verbose:
                print('output:', p.stdout.decode('ascii'))
                print('error:', p.stderr.decode('ascii'))


def main():
    job_list = [Job('job 01', HIGH, 3, True,
                    ['MANSCI', 'MANSCI-GPU', 'COB', 'COB-GPU'],
                    'python mycode.py', 'module load python3.7'),
                Job('job 02', LOW, 2, False,
                    ['MANSCI', 'MANSCI-GPU'],
                    'python mycode_2.py'),
                ]

    scheduler = JobScheduler('myemail@gmail.com',
                             available_queues=['MANSCI', 'MANSCI-GPU', 'COB', 'COB-GPU', 'all'])

    scheduler.submit(job_list)


if __name__ == "__main__":
    main()
