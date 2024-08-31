/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.batchnorm.global;

import java.util.Arrays;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.nn.unit.Train2Eval;
import z.dragon.nn.unit.simple.affine.Affine;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * Global Square Batch Normalization.
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class GlobalSqBatchNorm extends Affine implements Train2Eval {
    private static final long serialVersionUID = 1L;
    
    protected boolean affine;
    protected float beta1;//expoential average of mean(X)
    protected float beta2, eps;//exponential average of mean(X^2)
    
    transient protected double expBeta1;//to corrrect the exponential average
    transient protected double expBeta2;//to corrrect the exponential average
      
    transient protected Parameter run_mean;
    transient protected Parameter run_sqmean;
    
    public GlobalSqBatchNorm(boolean inplace, boolean affine,
            float beta1, float beta2, float eps,
            int... feature_dim)
    {
        super(inplace, feature_dim);
        this.affine = affine;
        this.beta1 = beta1; expBeta1 = 1;
        this.beta2 = beta2; this.eps = eps; expBeta2 = 1;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final boolean affine() { return affine; }
    public GlobalSqBatchNorm affine(boolean flag) { affine = flag; return this; }
    
    public final float beta1() { return beta1; }
    public GlobalSqBatchNorm beta1(float beta1) { this.beta1 = beta1; return this; }

    public final float beta2() { return beta2; } 
    public GlobalSqBatchNorm beta2(float beta2) { this.beta2 = beta2; return this; }
    
    public final float eps() {return eps;}
    public GlobalSqBatchNorm eps(float eps) { this.eps = eps; return this; }
    
    public final double exp_beta1() { return expBeta1; }
    public final double exp_beta2() { return expBeta2; }
    
    @Override public GlobalSqBatchNorm weight(Tensor weight) { set_weight(weight); return this; }
    @Override public GlobalSqBatchNorm bias(Tensor bias)  { set_bias(bias); return this; }

    public Tensor run_mean() { return run_mean.ts(); }
    public GlobalSqBatchNorm run_mean(Tensor mean) { set_run_mean(mean); return this; }
    public void set_run_mean(Tensor mean) {
        if(Tensor.isNull(mean)) throw new NullPointerException(name + ": run_mean is null");
        if(!mean.dimEquals(param_dim)) throw new IllegalArgumentException(String.format(
                "%s: run_mean.dum { got %s } != param_dim { got %s }", 
                name, Arrays.toString(run_mean.dim()), Arrays.toString(param_dim)));
        if(run_mean != null) run_mean.delete();
        run_mean.tensor(mean);
    }
    
    public Tensor run_sqmean() { return run_sqmean.ts(); }
    public GlobalSqBatchNorm run_sqmean(Tensor sqmean) { set_run_sqmean(sqmean); return this; }
    public void set_run_sqmean(Tensor sqmean) {
        if(Tensor.isNull(sqmean)) throw new NullPointerException(name + ": sqmean is null");
        if(!sqmean.dimEquals(param_dim)) throw new IllegalArgumentException(String.format(
                "%s: run_mean.dum { got %s } != param_dim { got %s }",
                name, Arrays.toString(run_sqmean.dim()), Arrays.toString(param_dim)));
        if(run_sqmean != null) run_sqmean.delete();
        run_sqmean.tensor(sqmean);
    }
    
    @Override 
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", affine = ").append(affine);
        sb.append(", features = ").append(features);
        sb.append(", param_dim = ["); Vector.append(sb, param_dim); sb.append("] ");
        sb.append(", [beta1, beta2, eps] = [")
                .append(beta1).append(", ")
                .append(beta2).append(", ")
                .append(eps).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected InlineGlobalSqBatchNorm<?> create_unit_core() {
        if(ucm.count() >= 1) throw new RuntimeException("Normalization Cores can only be called once in a cycle");
        return new InlineGlobalSqBatchNorm<>(this);
    }
    
    @Override
    protected void __init__(Engine eg)  {
        Parameter.delete(A, B, run_mean, run_sqmean);//params are inited to match the lastDim of input
        run_mean = Parameter.virual(eg.zeros(param_dim));//perform indentity transform
        run_sqmean = Parameter.virual(eg.ones(param_dim));
        if(affine) {//perform indentity transform
           A = new Parameter(eg.ones(param_dim)).need_grads(true);
           B = new Parameter(eg.zeros(param_dim)).need_grads(true);
        }
        Parameter.sync(A, B, run_mean, run_sqmean);
    }
    
    protected boolean training = true;
    @Override public boolean training() {return training;}
    @Override public GlobalSqBatchNorm train() { this.training = true; return this; }
    @Override public GlobalSqBatchNorm eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: param & state">
    protected String run_mean_key() { return name + ".run_mean"; }
    protected String run_sqmean_key() { return name + ".run_sqmean"; } 
    protected String params_key() { return name + ".params"; }
    
    @Override
    public void params(ParamSet set) {
        set.add(A, B, run_mean, run_sqmean);
    }
    
    @Override
    public void param_map(ParamMap<String> map) {
        super.param_map(map);//put [A, B]
        map.put(run_mean_key(), run_mean);
        map.put(run_sqmean_key(), run_sqmean);
    }
    
    @Override
    public void state(State dic) {
        super.state(dic);//put[A, B]
        dic.put(run_mean_key(), run_mean.ts());
        dic.put(run_sqmean_key(), run_sqmean.ts());
        dic.put(params_key(), State.floats((float)expBeta1, (float)expBeta2));
    }
    
    @Override
    public void update_state(State dic, boolean partial) {
        super.update_state(dic, partial);//update [A, B]
        
        run_mean.ts().set(dic.get(run_mean_key()), partial, name + ": fail to update run_mean");
        run_sqmean.ts().set(dic.get(run_sqmean_key()), partial,  name + ": fail to update run_sqmean");

        StateValue params = dic.get(params_key());
        State.set(params, name + "fail to update for params", partial, ()->{
            float[] arr = Vector.to_float_vector(params.toStringLines(), 2);
            expBeta1 = arr[0]; expBeta2 = arr[1];
        });
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineGlobalSqBatchNorm">
    public static class InlineGlobalSqBatchNorm<T extends GlobalSqBatchNorm> extends InlineAffine<T>
    {
        public InlineGlobalSqBatchNorm(T unit) { super(unit); }
        
        public boolean affine() { return ut.affine; }
        public float beta1() { return ut.beta1; }
        public float beta2() { return ut.beta2; } 
        public float eps() {return ut.eps;}
        public final double exp_beta1() { return ut.expBeta1; }
        public final double exp_beta2() { return ut.expBeta2; }
          
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        protected Tensor __forward_evaluate__(Engine eg, Tensor X, boolean inplace) {
            return (ut.affine?
                    eg.sqBatchNorm(inplace, X, ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps, ut.A.ts(), ut.B.ts()) : 
                    eg.sqBatchNorm(inplace, X, ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps));
        }
    
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace)  {
            if(!ut.training) return __forward_evaluate__(eg, X, inplace);
        
            //====[Stage1: update run_mean and run_sqmean]======================
            Tensor[] stats = eg.field_mean_sqmean(X, ut.features);
            Tensor dX_mean   = stats[0];
            Tensor dX_sqmean = stats[1];

            //Update: run_mean = a1*run_mean + a2*dX_mean
            //[1] a1 = alpha * (1 - alpha^(t-1)) / (1 - alpha^t)
            //[2] a2 = (1 - alpha) / (1 - alpha^t)
            double last_correct_beta1 = 1 - ut.expBeta1; 
            ut.expBeta1 *= ut.beta1;
            double corrrect_beta1 = 1 - ut.expBeta1;
            float a1 = (float) (ut.beta1 * last_correct_beta1 / corrrect_beta1);
            float a2 = (float) ((1 - ut.beta1) / corrrect_beta1);
            eg.add(true, a1, ut.run_mean.ts(), a2, dX_mean.c());//inplace: run_mean

            //Update: run_sqmean = b1*run_sqean + b2*dX_sqmean
            //[1] b1 = beta * (1 - beta^(t-1)) / (1 - beta^t)
            //[2] b2 = (1 - beta) / (1 - beta^t)
            double last_correct_beta2 = 1 - ut.expBeta2;
            ut.expBeta2 *= ut.beta2;
            double correct_beta2 = 1 - ut.expBeta2;
            float b1 = (float) (ut.beta2 * last_correct_beta2 / correct_beta2);
            float b2 = (float) ((1 - ut.beta2) / correct_beta2);
            eg.add(true, b1, ut.run_sqmean.ts(), b2, dX_sqmean.c());//inplace: run_sqmean
        
            //====[Stage2: Global Square Batch Normalization]=======================
            ut.run_mean.c(); ut.run_sqmean.c();
            Tensor Y = (ut.affine? 
                    eg.sqBatchNorm(inplace, X, ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps, ut.A.ts(), ut.B.ts()):
                    eg.sqBatchNorm(inplace, X, ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps));
            dX_mean.delete(); dX_sqmean.delete();//intermediate variables are not needed
            return Y;
        }
        //</editor-fold>
    
        //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
        protected Tensor __backward_no_affine__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdX()? //V1: Y is not changed / V2: X is not changed
                    eg.sqBatchNorm_deltaX_v2(grad_inplace, deltaY, holdX(), ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps) :
                    eg.sqBatchNorm_deltaX_v1(grad_inplace, deltaY, holdY(), ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps));
        }
    
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!ut.affine) return __backward_no_affine__(eg, deltaY, grad_inplace, backward_grads);//affine = false
     
            //======[integrally finds gradients for {A, B, X} ]=================
            if(ut.A.need_grads() && ut.B.need_grads() && backward_grads) {
                Tensor[] grads = (is_holdX() ? //V2: X is not changed / V1: Y is not changed
                        eg.sqBatchNorm_gradients_v2(grad_inplace, deltaY, holdX(), ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps, ut.A.ts()):
                        eg.sqBatchNorm_gradients_v1(grad_inplace, deltaY, holdY(), ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps, ut.A.ts(), ut.B.ts()));
            
                Tensor deltaX = grads[0];
                ut.A.grads().add(grads[1]);
                ut.B.grads().add(grads[2]);
            
                ut.A.ts().follow(deltaX);//When compute deltaX, A can't be changed
                if(is_holdY()) ut.B.ts().follow(deltaX);//When holdY & compute deltaX, B can't be changed
                return deltaX;
            }
        
            //======[separately finds gradients for {A, B, X} ]=================
            Tensor deltaA = null, deltaB = null, deltaX = null; 
            int gc_count = 0;
        
            if(ut.A.need_grads() && ut.B.need_grads()) {
                Tensor[] grads = (is_holdY() ?
                        eg.sqBatchNorm_deltaAB_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts()) ://V1: Y is not changed
                        eg.sqBatchNorm_deltaAB_v2(deltaY, holdX(), ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps));//V2: X is not changed
                
                ut.A.grads().add(deltaA = grads[0]);
                ut.B.grads().add(deltaB = grads[1]);
                if(grad_inplace) gc_count += 2;
            }
            else if(ut.A.need_grads()) {//B.need_grads = false
                deltaA = (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                        eg.sqBatchNorm_deltaA_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts()) :
                        eg.sqBatchNorm_deltaA_v2(deltaY, holdX(), ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps));
                ut.A.grads().add(deltaA);
                if(grad_inplace) gc_count++;
            }
            else if(ut.B.need_grads()) {//A.need_grads = false
                deltaB = eg.field_sum(deltaY, ut.features);
                ut.B.grads().add(deltaB);
                if(grad_inplace) gc_count++;
            }
        
            if(backward_grads) {
                deltaX = (is_holdX() ? //V2: X is not changed / V1: Y is not changed);
                        eg.sqBatchNorm_deltaX_v2(false, deltaY, holdX(), ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps, ut.A.ts()) :
                        eg.sqBatchNorm_deltaX_v1(false, deltaY, holdY(), ut.run_mean.ts(), ut.run_sqmean.ts(), ut.eps, ut.A.ts(), ut.B.ts()));
                
                ut.A.ts().follow(deltaX);//When compute deltaX, A can't be changed
                if(is_holdY()) ut.B.ts().follow(deltaX);//When holdY & compute deltaX, B can't be changed
                if(grad_inplace) gc_count++;
        }
        
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
