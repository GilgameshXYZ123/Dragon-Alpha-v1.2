/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual;

import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.unit.Unit;

/**
 * two input, and two output.
 * @author Gilgamesh
 */
public abstract class DualUnit extends Unit {
    private static final long serialVersionUID = 1L;
 
    @Override public boolean is_complex() { return false; }
    @Override public int input_tensor_num() { return 2; }
    @Override public int output_tensor_num() { return 1; }
    @Override protected abstract DualCore<?> create_unit_core();
}
