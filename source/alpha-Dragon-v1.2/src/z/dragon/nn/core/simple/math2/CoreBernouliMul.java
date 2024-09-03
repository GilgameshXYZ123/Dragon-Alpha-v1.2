/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleInplaceCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreBernouliMul<T extends SimpleUnit> extends SimpleInplaceCore<T> {
    protected boolean training = true;
    protected float p, v1, v2;
    
    transient protected Tensor R;
    
    public CoreBernouliMul(T unit, boolean inplace, 
            boolean training, float p, float v1, float v2) {
        super(unit, inplace);
        this.training = training;
        this.p = p;
        this.v1 = v1;
        this.v2 = v2;
    }
    
    public final float expect() { return v1*p + v2*(1.0f - p); } 
    public final float p() { return p; }
    public final float v1() { return v1; }
    public final float v2() { return v2; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    public void gc() {
        super.gc(); 
        if(R != null) { R.delete(); R = null; }
    }
    
    @Override
    public void variables(Tensor.TensorSet set) {
        super.variables(set);
        set.add(R);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        if(!training) return eg.linear(inplace, expect(), X, 0);//exp = (v1*p + v2*(1.0f - p))
        
        Tensor[] outs = eg.bernouli_mul(X, p, v1, v2);
        R = outs[1];
        return outs[0];//Y = outs[0]
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
