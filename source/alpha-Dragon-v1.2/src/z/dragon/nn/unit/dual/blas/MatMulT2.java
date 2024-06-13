/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.blas;

import z.dragon.nn.unit.dual.DualFunction;
import z.dragon.nn.core.dual.blas.CoreMatMulT2;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class MatMulT2 extends DualFunction
{
    private static final long serialVersionUID = 1L;

    @Override
    protected CoreMatMulT2<?> create_unit_core() {
        return new CoreMatMulT2<>(this);
    }
}
