""" 

"""
from pyStruct.database.signacSelector import SignacDatabaseSelector, get_signac_parent_job
from pyStruct.database.signacJobData import SignacJobData
from pyStruct.database.signacJobData import SignacJobData, CfdDataNotExist, TimeSeriesDataNotExist, SnapshotMatrixFileNotFoundError
from pyStruct.sampleCollector.sampleStructure import BoundaryCondition, Sample


def initialize_sample_from_signac(config) -> list[Sample]:

    # Filter job get JobData
    db = SignacDatabaseSelector(
        workspace = config.processor_workspace
    )
    functions = [
        lambda job: job.sp.proc_sp.method == config.method,
        lambda job: job.sp.proc_sp.columns == ','.join(config.columns)
    ]
    jobs = db.filter(functions)

    assert len(jobs) > 0, "No job is found from the given function conditions"

    samples = []
    for job in jobs:
        # Load JobData
        try:
            job_data = SignacJobData(
                job, 
                parent_job=get_signac_parent_job(config.parent_workspace, job, job.sp.cfd_sp),
                bc_mapper=dict(config.bc_mapper)
                )
            signac_bc = job_data.bc # without theta

            for theta_deg in config.theta_degs:
                bc = BoundaryCondition(
                    m_c = signac_bc['m_c'],
                    m_h = signac_bc['m_h'],
                    T_c = signac_bc['T_c'],
                    T_h = signac_bc['T_h'],
                    theta_deg = theta_deg
                )
                sample = Sample(
                        bc= bc,
                        N_dim=len(config.columns),
                        N_t=config.N_t,
                        dt=config.dt,
                        svd_truncate=config.svd_truncate,
                        coord=job_data.coord,
                        wall_true=job_data.T_wall(theta_deg, config.normalize_y)[-config.N_t:],
                        X_matrix_path = job_data.X_matrix_path
                    )
                # print(f'sample found from job: {job}')
                samples.append(sample)
        except CfdDataNotExist:
            pass
    return samples