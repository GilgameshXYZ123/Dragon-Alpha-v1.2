/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.blas;

import z.dragon.nn.core.dual.blas.CoreMatMul;
import z.dragon.nn.unit.dual.DualFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class MatMul extends DualFunction {
    private static final long serialVersionUID = 1L;
    
    @Override
    protected CoreMatMul<?> create_unit_core() { 
        return new CoreMatMul<>(this);
    }
}
