/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.affine;

import java.util.Arrays;
import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.dragon.nn.unit.simple.SimpleInplaceUnit;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * <pre>
 * read X from the last layer.
 * alloc: A, B
 * compute: Y(the next layer will read)
 * 
 * read deltaY from the next layer:
 * alloc: deltaW, deltaX(is need)
 * compute: deltaX(the last layer will read)
 * 
 * forward: Y = X(*)A + B #element multiply
 * back: deltaX = deltaY*W^T
 * </pre>
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Affine extends SimpleInplaceUnit {
    private static final long serialVersionUID = 1L;
    
    protected int features;
    protected int[] param_dim;//[...., mem_width = input.lastDim]
    
    transient protected Parameter A;
    transient protected Parameter B;
    
    public Affine(boolean inplace, int... feature_dim) {
        super(inplace);

        if(feature_dim == null || feature_dim.length == 0)
            throw new NullPointerException("feature_dim is null");
        
        if(feature_dim.length == 1) {//input feature is 2D
            this.features = feature_dim[0];
            this.param_dim = new int[]{ feature_dim[0] };
        }
        else {//input feature >= 3D. to save memory: the param is 2D[mem_height, mem_width]
            this.features = Vector.mul(feature_dim);
            int lastDim = feature_dim[feature_dim.length - 1];
            this.param_dim = new int[]{ features / lastDim, lastDim };
        }
    }
     
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final int features() { return features; }
    public final int[] param_dim() { return param_dim; }
    
    public Parameter weight_param() { return A; }
    public Tensor weight() { return A.ts(); }
    public Affine weight(Tensor weight) { set_weight(weight); return this; }
    protected void set_weight(Tensor weight) {
        if(Tensor.isNull(weight)) throw new NullPointerException(name + ": weight is null");
        if(!weight.dimEquals(param_dim)) throw new IllegalArgumentException(String.format(
                "%s: weight.dim { got %s } != param_dim { got %s }",
                name, Arrays.toString(weight.dim()), Arrays.toString(param_dim)));
        if(A != null) A.delete();
        A.tensor(weight);
    }
    
    public Parameter bias_param() { return B; }
    public Tensor bias() { return B.ts(); }
    public Affine bias(Tensor bias) { set_bias(bias); return this; }
    protected void set_bias(Tensor bias) {
        if(Tensor.isNull(bias)) throw new NullPointerException(name + ": bias is null");
        if(!bias.dimEquals(param_dim)) throw new IllegalArgumentException(String.format(
                "%s: bias.dim { got %s } != param_dim { got %s }",
                name, Arrays.toString(bias.dim()), Arrays.toString(param_dim)));
        if(B != null) B.delete();
        B.tensor(bias);
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", features = ").append(features);
        sb.append(", param_dim = ["); Vector.append(sb, param_dim); sb.append("] ");
        sb.append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: init"> 
    @Override
    protected InlineAffine<?> create_unit_core() {
        return new InlineAffine<>(this);
    }
    
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(A, B);//params are inited to match the lastDim of input
        A = new Parameter(eg.ones(param_dim)).need_grads(true);//perform indentity transform
        B = new Parameter(eg.zeros(param_dim)).need_grads(true);
        Parameter.sync(A, B);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: param & state">
    protected String weight_key() { return name + ".weight"; }
    protected String bias_key() { return name + ".bias"; }
    
    @Override 
    public void params(ParamSet set) { 
        set.add(A, B);//null is not include
    }
    
    @Override
    public void param_map(ParamMap<String> map) {
        map.put(weight_key(), A);//null is not include
        map.put(bias_key(), B);//null is not include
    }
    
    @Override 
    public void state(State dic) {
        dic.put(weight_key(), A.ts());//null is not include
        dic.put(bias_key(), B.ts());//null is not include
    }
    
    @Override
    public void update_state(State dic, boolean partial)  {
        if(A != null) A.ts().set(dic.get(weight_key()), partial, name + ": fail to update state for weight");
        if(B != null) B.ts().set(dic.get(bias_key()), partial, name + ": fail to update state for bias");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAffine">
    public static class InlineAffine<T extends Affine> extends SimpleInplaceInline<T> {
        public InlineAffine(T unit) { super(unit); }
        
        public int features() { return ut.features; }
        public int[] param_dim() { return ut.param_dim; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation"> 
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.affine(inplace, X, ut.A.ts(), ut.B.ts());
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            Tensor deltaA = null, deltaB = null, deltaX = null;
            int gc_count = 0;
        
            if(ut.A.need_grads() && ut.B.need_grads()) {//A.need_grads = B.need_grads = true
                Tensor[] delta = (is_holdY()? 
                        eg.affine_deltaAB_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts())://V1: Y is not changed
                        eg.affine_deltaAB_v2(deltaY, holdX(), ut.features));//V2: X is not changed
            
                deltaA = delta[0]; ut.A.grads().add(deltaA);
                deltaB = delta[1]; ut.B.grads().add(deltaB);
                if(grad_inplace) gc_count += 2;
            }
            else if(ut.A.need_grads()) {//B.need_grads = false
                deltaA = (is_holdY()?
                        eg.affine_deltaA_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts())://V1: Y is not changed
                        eg.affine_deltaA_v2(deltaY, holdX(), ut.features));//V2: X is not changed
                ut.A.grads().add(deltaA);
                if(grad_inplace) gc_count++;
            }
            else if(ut.B.need_grads()) {//A.need_grads = false
                deltaB = eg.field_sum(deltaY, ut.features);
                ut.B.grads().add(deltaB);
                if(grad_inplace) gc_count++;
            }
        
            if(backward_grads) {
                deltaX = eg.mul_row(false, deltaY, ut.A.ts());//false: deltaY is read only
                ut.A.ts().follow(deltaX);//When compute deltaX, A can't be changed
                if(grad_inplace) gc_count++;
            }
        
            //the final gc process----------------------------------------------
            if(gc_count != 0) {//when deltaA, deltaB, deltaX are cauculated, deltaY is not needed
                CountGc gc = new CountGc(gc_count, deltaY);
                if(deltaA != null) { deltaA.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
            }
            
            return deltaX;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
