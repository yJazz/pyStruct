from dataclasses import dataclass
import numpy as np
from pyStruct.sampleCollector.sampleStructure import Sample, BoundaryCondition
from pyStruct.sampleCollector.sampleSetStructure import SampleSet

class TestBoundarCondition:
    def test_1_test_eq(self):
        bc_1 = BoundaryCondition(m_c=1, m_h=1, T_c=1, T_h=1, theta_deg=0)
        bc_2 = BoundaryCondition(m_c=1, m_h=1, T_c=1, T_h=1, theta_deg=0)
        assert (bc_1 == bc_2) 

        # small floating error, still same
        bc_3 = BoundaryCondition(m_c=1.00001, m_h=1, T_c=1, T_h=1, theta_deg=0)
        assert (bc_1 == bc_3)

        # different bc
        bc_4 = BoundaryCondition(m_c=5, m_h=3, T_c=1, T_h=1, theta_deg=0)
        assert (bc_1 == bc_4) is False

class TestPodSample:
    def setup_method(self):
        self.bc = BoundaryCondition(
            m_c=300.1, 
            m_h=3.21, 
            T_c= 70.7, 
            T_h=30.0,
            theta_deg=0.0
            )
        self.loc_index = 1584
        self.wall_true = np.random.rand(1000)
        