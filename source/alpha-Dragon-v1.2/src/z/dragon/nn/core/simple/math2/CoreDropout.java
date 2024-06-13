/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.core.simple.SimpleInplaceCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreDropout<T extends SimpleUnit> extends SimpleInplaceCore<T>
{
    protected boolean training = true;
    protected float p;
    
    transient protected Tensor R;
    
    public CoreDropout(T unit, boolean inplace,
            boolean training, float nonzero_p) {
        super(unit, inplace);
        this.training = training;
        this.p = nonzero_p;
    }
    
    public final float nonzero_percent() { return p; }
    public final boolean training() { return training; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    public void gc() {
        super.gc(); 
        if(R != null) { R.delete(); R = null; }
    }
    
    @Override
    public void variables(TensorSet set) {
        super.variables(set);
        set.add(R);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        if(!training) return (inplace? X : eg.copy(X));//exp = (1/p)*p + (1 - p)*0

        Tensor[] outs = eg.dropout(X, p);
        R = outs[1];//R = eg.Bernouli(p, pr, 0, X.dim());
        return outs[0];//Y = outs[0] = g.mul(inplace, X, R.c());
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads ? //when deltaX is cauculated, rber is not needed
                eg.mul(grad_inplace, deltaY, R).dual(()->{ R.delete(); }):
                null);
    }
    //</editor-fold>
}
