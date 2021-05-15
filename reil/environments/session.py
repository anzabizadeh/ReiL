from __future__ import annotations

import multiprocessing
from multiprocessing.context import BaseContext
from typing import List, Literal, Optional, Union

from reil.environments import Task

Task_n_Session = Union[Task, "Session"]


class Session:
    def __init__(
            self, name: str, path: str, main_task: Task_n_Session,
            tasks_before: Optional[List[Task_n_Session]] = None,
            tasks_after: Optional[List[Task_n_Session]] = None,
            tasks_before_iteration: Optional[List[Task_n_Session]] = None,
            tasks_after_iteration: Optional[List[Task_n_Session]] = None,
            separate_process: Optional[List[Literal[
                'tasks_before',
                'tasks_after',
                'tasks_before_iteration',
                'tasks_after_iteration']]] = None,
            process_type: Optional[Literal[
                'spawn', 'fork', 'forkserver']] = None):
        self.iterations = 1  # to be consistent with `Task`
        self._name = name
        self._path = path
        self._main_task = main_task
        self._tasks_before = tasks_before
        self._tasks_after = tasks_after
        self._tasks_before_iteration = tasks_before_iteration
        self._tasks_after_iteration = tasks_after_iteration
        self._separate_process = separate_process or []
        if separate_process:
            self._process_type = process_type or 'spawn'
        else:
            self._process_type = None

    @staticmethod
    def _run_tasks(
            task_list: Optional[List[Task_n_Session]], iteration: int,
            separate_process: bool,
            context: Optional[BaseContext] = None):
        p = None
        if task_list:
            for t in task_list:
                if separate_process:
                    p = context.Process(target=t.run, args=(iteration,))
                    p.start()
                else:
                    t.run(iteration)
        # if p:
        #     p.join()

    def run(self, iteration: int = None):
        '''Run the session

        Parameters
        ----------
        iteration : None, optional
            iteration is used so that the signature is compatible with
            `Task`. This allows having a `Session` as part of
            another `Session`.
        '''
        context = (multiprocessing.get_context(self._process_type)
                   if self._separate_process else None)

        self._run_tasks(
            self._tasks_before, iteration or 0,
            'tasks_before' in self._separate_process, context)

        itr = iteration or 0
        for itr in range(iteration or 0, self._main_task.iterations):
            self._run_tasks(
                self._tasks_before_iteration, itr,
                'tasks_before_iteration' in self._separate_process, context)

            self._main_task.run(itr)

            self._run_tasks(
                self._tasks_after_iteration, itr,
                'tasks_after_iteration' in self._separate_process, context)

        self._run_tasks(
            self._tasks_after, itr + 1,
            'tasks_after' in self._separate_process, context)


# if __name__ == '__main__':
#     from reil.agents.agent import Agent
#     from reil.learners import Learner
#     from reil.datatypes.interaction_protocol import (
#         Entity, InteractionProtocol)
#     from reil.learners.learning_rate_schedulers import ConstantLearningRate
#     from reil.subjects.subject import Subject
#     s = Session(
#         name='test', path='.',
#         main_task=Task(
#             'main', '.', 'training',
#             InteractionProtocol(
#                 Entity('agent'), Entity('subject'),
#                 'default', 'default', 'default', 1, 'iteration'),
#             Agent(Learner(ConstantLearningRate(0.01)), None),
#             Subject(),
#             False),
#         tasks_before=[],
#         tasks_after=[],
#         separate_process=[]
#     )

#     s.run()
