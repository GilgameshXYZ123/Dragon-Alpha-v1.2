/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.layernorm;

import java.util.Arrays;
import z.dragon.engine.Counter;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.affine.Affine;

/**
 *
 * @author Gilgamesh
 */
public class GroupNorm extends Affine {
    private static final long serialVersionUID = 1L;
    
    protected boolean affine;
    protected int group;
    protected float eps;
    transient protected Tensor X_mean;
    transient protected Tensor X_sqmean;

    public GroupNorm(boolean inplace, boolean affine, 
            float eps, int group, int channel)
    {
        super(inplace, channel);
        if (channel % group != 0) throw new IllegalArgumentException(String.format(
                "channel { got %d } mod group { got %d } != 0", channel ,group));
        this.group = group;
        this.affine = affine;
        this.eps = eps;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int group() { return group; }
    
    public final boolean affine() { return affine; }
    public GroupNorm affine(boolean flag) { affine = flag; return this; }
     
    public final float eps() { return eps; }
    public GroupNorm eps(float eps) { this.eps = eps; return this;}
    
    @Override public GroupNorm weight(Tensor weight) { super.weight(weight); return this; }
    @Override public GroupNorm bias(Tensor bias) { super.bias(bias); return this; }
    
    public Tensor mean() { return X_mean; }
    public Tensor sqmean() { return X_sqmean; }
    
    @Override 
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", affine = ").append(affine);
        sb.append(", [feature_num, param_dim] = [ ")
                .append(features).append(", ")
                .append(Arrays.toString(param_dim)).append(" ]");
        sb.append(", eps = ").append(eps).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected InlineGroupNorm<?> create_unit_core() {
        if(ucm.count() >= 1) throw new RuntimeException("Normalization Cores can only be called once in a cycle");
        return new InlineGroupNorm<>(this);
    }
    
    @Override
    protected void __init__(Engine eg) {
        eg.delete(A, B);//params are inited to match the lastDim of input
        if(affine) {//perform indentity transform
            A = new Parameter(eg.ones(param_dim)).need_grads(true);
            B = new Parameter(eg.zeros(param_dim)).need_grads(true);
            Parameter.sync(A, B);
        }
    }
    
    @Override
    public void variables(Tensor.TensorSet set) {
        super.variables(set);
        set.add(X_mean, X_sqmean);
    }
    
    @Override
    public void gc() {
        super.gc();
        if(X_mean != null) { X_mean.delete(); X_mean = null; }
        if(X_sqmean != null) { X_sqmean.delete(); X_sqmean = null; }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: GroupNorm_builtin">
    public static class InlineGroupNorm<T extends GroupNorm>  extends InlineAffine<T> {
        public InlineGroupNorm(T unit) { super(unit); }
        
        public final boolean affine() { return ut.affine; }
        public final float eps() { return ut.eps; }
          
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
           return null;
        }
        //</editor-fold>
    
        //<editor-fold defaultstate="collapsed" desc="running-area: backwards-propagation">
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return null;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
