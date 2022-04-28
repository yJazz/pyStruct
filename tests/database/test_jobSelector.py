from pyStruct.database.signacSelector import SignacDatabaseSelector, OpenSignacProjectFails

class TestSignacjobSelector:
    @classmethod
    def setup_class(cls):
        cls.processor_workspace = r'F:\project2_phD_bwrx\db_bwrx_processor'
        cls.parent_workspace = r'F:\project2_phD_bwrx\db_bwrx_cfd'
        cls.columns= ['Velocity[i] (m/s)','Velocity[j] (m/s)','Velocity[k] (m/s)']
        cls.N_t= 1000
        cls.mapper = {
            'M_COLD_KGM3S': 'm_c', 
            'M_HOT_KGM3S': 'm_h',
            'T_COLD_C': 'T_c',
            'T_HOT_C': 'T_h'
        }


    def test_1_open_project_if_project_exist(self):
        db = SignacDatabaseSelector(
            workspace = TestSignacjobSelector.processor_workspace
        )
        assert hasattr(db, '_project')
    def test_open_project_fail(self):
        try:
            db = SignacDatabaseSelector(
                workspace = '.',
            )
        except OpenSignacProjectFails:
            pass
        except:
            assert False
    
    def test_filter_jobs_by_valid_functions(self):
        db = SignacDatabaseSelector(
            TestSignacjobSelector.processor_workspace 
            )
        N_init = len(db.jobs)
  
        functions = [
            lambda job: job.sp.proc_sp.method =='dmd', # dmd only
            lambda job: job.sp.proc_sp.columns == ','.join(TestSignacjobSelector.columns) 
        ]
        jobs = db.filter(functions)
        assert N_init > len(jobs)

    def test_filter_jobs_by_invalid_functions(self):
        db = SignacDatabaseSelector(
            TestSignacjobSelector.processor_workspace
            )

        functions = ['a', 'b']
        try: 
            db.filter(functions)
        except AssertionError:
            pass
        except:
            assert False
        




    

            



        




