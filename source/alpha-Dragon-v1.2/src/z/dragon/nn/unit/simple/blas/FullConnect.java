/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.blas;

import java.util.Arrays;
import z.dragon.common.state.State;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.nn.unit.simple.SimpleUnit;
import z.dragon.nn.core.simple.SimpleCore;
import z.util.lang.annotation.Passed;

/**
 * <pre>
 * read X from the last layer.
 * alloc: W, B
 * compute: Y(the next layer will read)
 * 
 * read deltaY from the next layer:
 * alloc: deltaW, deltaX(is need)
 * compute: deltaX(the last layer will read)
 * 
 * forward: Y = X*W + B 
 * back: deltaX = deltaY*W^T
 * </pre>
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class FullConnect extends SimpleUnit {
    private static final long serialVersionUID = 562781240130001L;
    
    protected int IW, OW;
    protected boolean biased;
    
    transient protected Parameter W;
    transient protected Parameter B;
    
    public FullConnect(boolean biased, int in_features, int out_features) {
        this.biased = biased;
        this.IW = in_features;
        this.OW = out_features;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final boolean biased() { return biased; }
    public final int in_features() { return IW; }
    public final int out_features() { return OW; }
    
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public FullConnect pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    public final int[] fans() { return new int[]{ IW, OW }; }
    
    public Parameter weight_param() { return W; }
    public Tensor weight() { return W.ts(); }
    public FullConnect weight(Tensor weight) { 
        if(Tensor.isNull(weight)) throw new NullPointerException("weight is null");
        if(!weight.dimEquals(IW, OW)) throw new IllegalArgumentException(String.format(
                "%s: weight.dim { got %s } != [IW, OW]", 
                name, Arrays.toString(weight.dim()), IW, OW));
        if(W != null) W.delete(); 
        W.tensor(weight);
        return this; 
    }
    
    public Parameter bias_param() { return B; }
    public Tensor bias() { return B.ts(); }
    public FullConnect bias(Tensor bias) {
        if(Tensor.isNull(bias)) throw new NullPointerException("bias is null");
        if(!bias.dimEquals(OW)) throw new IllegalArgumentException(String.format(
                "%s: bias.dim { got %s } != [OW] = [%d]",
                name, Arrays.toString(bias.dim()), OW));
        if(B != null) B.delete(); 
        B.tensor(bias);
        return this;
    }
    
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { biased = ").append(biased);
        sb.append(", [in_features, out_features] = [").append(IW).append(", ").append(OW).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    transient private boolean baseW = true;//is the first gradient of W
    transient private boolean baseB = true;//is the first gradient of B
    
    protected final synchronized void reset_backward() { baseW = baseB = true;  }
    protected final synchronized boolean baseW() {  boolean old = baseW; baseW = false; return old; }
    protected final synchronized boolean baseB() { boolean  old = baseB; baseB = false; return old; }
    
    @Override
    protected InlineFullConnect create_unit_core() {
        return new InlineFullConnect(this);
    }
    
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(W, B);
       
        Tensor tW = eg.empty(IW, OW);
        W = new Parameter(tW).need_grads(true);
        eg.kaiming_uniform(tW.c(), fans(), (float)Math.sqrt(5.0));//inplace: tW
        
        if(biased) {
            float bound = (float) (1.0 / Math.sqrt(IW));
            Tensor tB = eg.uniform(-bound, bound, OW);
            B = new Parameter(tB).need_grads(true);
        }
        
        Parameter.sync(W, B);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: param & state">
    protected String weight_key() { return name + ".weight"; }
    protected String bias_key() { return name + ".bias"; }
    
    @Override
    public void params(ParamSet set) {
        set.add(W);
        if(biased) set.add(B);
    }
    
    @Override
    public void param_map(ParamMap<String> map) {
        map.put(weight_key(), W);
        if(biased) map.put(bias_key(), B);
    }
    
    @Override
    public void state(State dic) {
        dic.put(weight_key(), W.ts());
        if(biased) dic.put(bias_key(), B.ts());
    }
    
    @Override
    public void update_state(State dic, boolean partial) {
        W.ts().set(dic.get(weight_key()), partial, name + ": fail to update state for weight");
        if(biased) B.ts().set(dic.get(bias_key()), partial, name + ": fail to update state for bias");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineFullConnect"> 
    public static class InlineFullConnect extends SimpleCore<FullConnect> {
        transient private Tensor y;
        
        public InlineFullConnect(FullConnect unit) { super(unit); }
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public boolean biased() { return ut.biased; }
        public int in_features() { return ut.IW; }
        public int out_features() { return ut.OW; }
        
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected void __before_forward__(Engine eg, Tensor X) { 
            ut.reset_backward();
            y = (ut.pre_alloc_forward ? //X[N, IW] -> Y[N, OW]
                    eg.alloc.fullconnect(X, ut.W.ts()) : 
                    null);
        }
        
        @Override
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(ut.pre_alloc_forward) {
                Tensor out = y.c(); y = null;
                return (ut.biased ? //X[N, IW] -> Y[N, OW]
                        eg.fullconnect_biased(out, X, ut.W.ts(), ut.B.ts()) :
                        eg.fullconnect(out, X, ut.W.ts()));
            }
            
            return (ut.biased ? //X[N, IW] -> Y[N, OW]
                    eg.fullconnect_biased(X, ut.W.ts(), ut.B.ts()) :
                    eg.fullconnect(X, ut.W.ts()));
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running area: backward-propagation">
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY,
              boolean grad_inplace, boolean backward_grads) 
        {
           Tensor deltaW = null, deltaB = null, deltaX = null;
           int gc_count = 0;
        
           if(ut.W.need_grads()) {//find the gradient of weight: [K, N] * [N, M] -> [K, M]
               deltaW = eg.fullconnect_deltaW(holdX(), deltaY);
               ut.W.accumulate(ut.baseW(), deltaW);
               if(grad_inplace) gc_count++;
            }
        
            if(ut.biased && ut.B.need_grads()) {//find the gradient of bias
                deltaB = eg.field_sum(deltaY, ut.OW);
                ut.B.accumulate(ut.baseB(), deltaB);
                if(grad_inplace) gc_count++;
            }
        
            if(backward_grads) {//find the gradient of-input features
                deltaX = eg.fullconnect_deltaX(deltaY, ut.W.ts());
                ut.W.ts().follow(deltaX);//When compute deltaX, W can't be changed
                if(grad_inplace) gc_count++;
            }
        
            if(gc_count != 0) {//when deltaW, deltaB, deltaX are cauculated, deltaY is not needed
                CountGc gc = new CountGc(gc_count, deltaY);
                if(deltaW != null) { deltaW.dual(()-> { gc.countDown() ;}).remote_sync(); }
                if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
            }
        
            return deltaX;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
