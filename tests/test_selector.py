from msilib.schema import Error
import pytest
from pyStruct.data.selector import DatabaseSelector
from signac.contrib.job import Job

@pytest.fixture
def db():
    return DatabaseSelector(database_path= r'F:\project2_phD_bwrx\db_bwrx_processor')
    

def test_1_filter_jobs(db):
    # Success Filter 
    no_stress = lambda job: job.sp.cfd_sp.model.STRESS_ON == False
    jobs = db.filter_jobs(db.jobs, filter_fn=no_stress)
    assert isinstance(jobs[0], Job)
    assert len(jobs) <= len(db.jobs)

    # Failed Filter
    not_valid_function = 'a'
    try:
        filtered_jobs = db.filter_jobs(db.jobs, filter_fn=not_valid_function)
    except AssertionError:
        print("The failed function is captured")
    except:
        assert False


def test_2_filter_pipe(db) :
    columns = 'Velocity[i] (m/s),Velocity[j] (m/s),Velocity[k] (m/s),Temperature (K)'
    method = 'pod'
    no_stress = lambda job: job.sp.cfd_sp.model.STRESS_ON == False
    pod_method = lambda job: job.sp.proc_sp.method == method
    pod_4d = lambda job: job.sp.proc_sp.columns == columns

    jobs = db.filter_pipe(db.jobs, no_stress, pod_method, pod_4d)
