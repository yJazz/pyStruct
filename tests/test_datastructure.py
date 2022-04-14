from pyStruct.machines.datastructures import BoundaryCondition

def test_1_test_eq():
    bc_1 = BoundaryCondition(m_c=1, m_h=1, T_c=1, T_h=1)
    bc_2 = BoundaryCondition(m_c=1, m_h=1, T_c=1, T_h=1)
    assert (bc_1 == bc_2) 

    # small floating error, still same
    bc_3 = BoundaryCondition(m_c=1.00001, m_h=1, T_c=1, T_h=1)
    assert (bc_1 == bc_3)

    # different bc
    bc_4 = BoundaryCondition(m_c=5, m_h=3, T_c=1, T_h=1)
    assert (bc_1 == bc_4) is False