/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.combiner;

import z.dragon.nn.core.combiner.CombinerCore;
import z.dragon.nn.unit.Unit;

/**
 *
 * @author Gilgamesh
 */
public abstract class Combiner extends Unit { 
    private static final long serialVersionUID = 1L;
    
    @Override public boolean is_complex() { return false; }
    @Override public int input_tensor_num() { return -1; }
    @Override public int output_tensor_num() { return 1; }
    @Override protected abstract CombinerCore<?> create_unit_core();
}
