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
public class CoreExpand<T extends SimpleUnit> extends SimpleInplaceCore<T>
{
    protected int[] start;
    protected int[] out_dim;
    
    transient protected int[] in_dim;
    
    public CoreExpand(T unit, boolean inplace, int[] start_point, int[] out_dim) {
        super(unit, inplace);
        this.start = start_point;
        this.out_dim = out_dim;
    }

    public final int[] start() { return start; }
    public final int[] out_dim() { return out_dim; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        in_dim = X.dim();
        return eg.expand(inplace, X, start, out_dim);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.crop(grad_inplace, deltaY, start, in_dim) : 
                null);
    }
    //</editor-fold>
}
