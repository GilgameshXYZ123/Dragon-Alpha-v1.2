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
public class CoreReshape<T extends SimpleUnit> extends SimpleInplaceCore<T> {
    protected int[] outDim;
    
    transient protected int[] inDim;
    
    public CoreReshape(T unit, boolean inplace, int...outDim) { 
        super(unit, inplace); 
        this.outDim = outDim;
    }
    
    public final int[] in_dim() { return inDim; }
    public final int[] out_dim() { return outDim; }
     
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        inDim = X.dim();
        return eg.reshape(inplace, X, outDim);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.reshape(grad_inplace, deltaY, inDim):
                null);
    }
    //</editor-fold>
}
