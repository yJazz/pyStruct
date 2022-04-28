from pyStruct.database.signacSelector import SignacDatabaseSelector, get_signac_parent_job
from pyStruct.database.signacJobData import SignacJobData


PROCESSOR_WORKSPACE = r'F:\project2_phD_bwrx\db_bwrx_processor'
PARENT_WORKSPACE = r'F:\project2_phD_bwrx\db_bwrx_cfd'
COLUMNS= ['Velocity[i] (m/s)','Velocity[j] (m/s)','Velocity[k] (m/s)']
N_T= 1000
D_T = 0.005
MAPPER = {
    'M_COLD_KGM3S': 'm_c', 
    'M_HOT_KGM3S': 'm_h',
    'T_COLD_C': 'T_c',
    'T_HOT_C': 'T_h'
}

if __name__ == "__main__":
    # Select Jobs from database Case
    functions = [
        lambda job: job.sp.proc_sp.method =='dmd', # dmd only
        lambda job: job.sp.proc_sp.columns == ','.join(COLUMNS) # three dim
    ]
    db = SignacDatabaseSelector(
        workspace = PROCESSOR_WORKSPACE
    )
    jobs  = db.filter(functions)

    # Data Initalization
    # Containing inputs like bc, X_matrix,... 
    job_datas = [SignacJobData(job, get_signac_parent_job(PARENT_WORKSPACE, job, job.sp.cfd_sp), bc_mapper=MAPPER) for job in jobs]

    # Process

