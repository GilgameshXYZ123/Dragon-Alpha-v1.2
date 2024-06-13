/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple;

import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.Unit;

/**
 * one input, and one output.
 * @author Gilgamesh
 */
public abstract class SimpleUnit extends Unit
{
    private static final long serialVersionUID = 56278124000000000L;
    
    @Override public boolean is_complex() { return false; }
    @Override public int input_tensor_num() { return 1; }
    @Override public int output_tensor_num() { return 1; }
    @Override protected abstract SimpleCore<?> create_unit_core();
}