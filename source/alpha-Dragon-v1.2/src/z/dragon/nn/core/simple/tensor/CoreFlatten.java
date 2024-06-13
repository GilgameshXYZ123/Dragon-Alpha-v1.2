/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.tensor;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleInplaceCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreFlatten<T extends SimpleUnit> extends SimpleInplaceCore<T>
{
    transient protected int[] inDim;
    transient protected int[] outDim;
    
    public CoreFlatten(T unit, boolean inplace) {
        super(unit, inplace);
    }
    
    public final int[] in_dim() { return inDim; }
    public final int[] out_dim() { return outDim; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        inDim = X.dim();
        outDim = new int[] { X.dim(0), X.length() / X.dim(0) };
        return eg.reshape(inplace, X, outDim);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.reshape(grad_inplace, deltaY, inDim) :
                null);
    }
    //</editor-fold>
}
