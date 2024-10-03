from utilities.Coordinator import Coordinator
from utilities.Helpers import Task

if __name__ == '__main__':
    master = Coordinator(Task.Task01, 100, [250, 700], 2)
    # master.start_task()

    master.change_task(Task.Task02)
    # master.start_task()

    master.change_task(Task.Task03)
    # master.start_task(1, 4, True)

    master.change_task(Task.Task04)
    # master.start_task()

    # master.change_task(Task.Task05)
    # master.start_task()

    master.change_task(Task.Task06)
    # master.start_task()

    master.change_task(Task.Task07)
    # master.start_task()

    master.change_task(Task.Task08)
    # master.start_task()

    master.change_task(Task.Task09)
    # master.start_task()

    master.change_task(Task.Task11)
    master.start_task()
