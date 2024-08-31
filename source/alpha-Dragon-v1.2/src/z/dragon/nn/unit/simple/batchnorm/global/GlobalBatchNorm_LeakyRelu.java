 /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.batchnorm.global;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * Global Batch Normalization.
 * Y = leakyRelu(global_batchNorm(X, A, B))
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class GlobalBatchNorm_LeakyRelu extends GlobalBatchNorm {
    private static final long serialVersionUID = 1L;
    
    protected float k;

    public GlobalBatchNorm_LeakyRelu(boolean inplace, boolean affine, 
            float beta1, float beta2, float eps, 
            float negative_slope,
            int... feature_dim) 
    {
        super(inplace, affine,
                beta1, beta2, eps, 
                feature_dim);
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final float negative_slop() { return k; }
    public GlobalBatchNorm_LeakyRelu negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    @Override public GlobalBatchNorm_LeakyRelu affine(boolean flag) { affine = flag; return this; }
    @Override public GlobalBatchNorm_LeakyRelu beta1(float beta1) { this.beta1 = beta1; return this; }
    @Override public GlobalBatchNorm_LeakyRelu beta2(float beta2) { this.beta2 = beta2; return this; }
    @Override public GlobalBatchNorm_LeakyRelu eps(float eps) { this.eps = eps; return this; }
   
    @Override public GlobalBatchNorm_LeakyRelu weight(Tensor weight) { set_weight(weight); return this; }
    @Override public GlobalBatchNorm_LeakyRelu bias(Tensor bias) { set_bias(bias); return this; }
      
    @Override public GlobalBatchNorm_LeakyRelu run_mean(Tensor mean) { set_run_mean(mean); return this; }
    @Override public GlobalBatchNorm_LeakyRelu run_var(Tensor var) { set_run_var(var); return this; }
    
    @Override public GlobalBatchNorm_LeakyRelu train() { this.training = true; return this; }
    @Override public GlobalBatchNorm_LeakyRelu eval() { this.training = false; return this; }
    
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
                .append(eps).append("]");
        sb.append(", negative_slope = ").append(k);
        sb.append(" }");
    }
    
    @Override
    protected InlineGlobalBatchNorm_LeakyRelu<?> create_unit_core() {
        return new InlineGlobalBatchNorm_LeakyRelu<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineGlobalBatchNorm_LeakyRelu">
    public static class InlineGlobalBatchNorm_LeakyRelu
            <T extends GlobalBatchNorm_LeakyRelu>extends InlineGlobalBatchNorm<T>
    {
        public InlineGlobalBatchNorm_LeakyRelu(T unit) { super(unit); }
        
        public final float negatice_slope() { return ut.k; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected Tensor __forward_evaluate__(Engine eg, Tensor X, boolean inplace) {
            return (ut.affine?
                    eg.batchNorm_leakyRelu(inplace, X, ut.run_mean.ts(), ut.run_var.ts(), ut.eps, ut.A.ts(), ut.B.ts(), ut.k) :
                    eg.batchNorm_leakyRelu(inplace, X, ut.run_mean.ts(), ut.run_var.ts(), ut.eps, ut.k));
        }
    
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) return __forward_evaluate__(eg, X, inplace);
        
            //====[Stage1: update run_mean and run_var]=========================
            Tensor[] stats = eg.field_var_mean(false, X, ut.features);
            Tensor dX_var  = stats[0];//unbiased = false
            Tensor dX_mean = stats[1];
        
            //Update: run_mean = a1*run_mean + a2*dX_mean
            //[1] a1 = alpha * (1 - alpha^(t-1)) / (1-alpha^t)
            //[2] a2 = (1 - alpha) / (1-alpha^t)
            double last_correct_beta1 = 1 - ut.expBeta1; 
            ut.expBeta1 *= ut.beta1;
            double corrrect_beta1 = 1 - ut.expBeta1;
            float a1 = (float) (ut.beta1 * last_correct_beta1 / corrrect_beta1);
            float a2 = (float) ((1 - ut.beta1) / corrrect_beta1);
            eg.add(true, a1, ut.run_mean.ts(), a2, dX_mean.c());//inplace: run_mean

            //Update: run_var = b1*run_var + b2*dX_var
            //[1] K = N / (N - 1), N = batch_size, for unbiased  estimation
            //[1] b1 = beta * (1-beta^(t-1)) / (1-beta^t)
            //[2] b2 = K * (1 - beta) / (1 - beta^t)
            int N = X.length() / ut.features;
            double K = N / (N - 1.0);
            double last_correct_beta2 = 1 - ut.expBeta2;
            ut.expBeta2 *= ut.beta2;
            double correct_beta2 = 1 - ut.expBeta2;
            float b1 = (float) (ut.beta2 * last_correct_beta2 / correct_beta2);
            float b2 = (float) (K * (1 - ut.beta2) / correct_beta2);
            eg.add(true, b1, ut.run_var.ts(), b2, dX_var.c());//inplace: run_var
        
            //====[Stage2: Global Batch Normalization]==========================
            ut.run_mean.c(); ut.run_var.c();
            Tensor Y = (ut.affine?
                    eg.batchNorm_leakyRelu(inplace, X, ut.run_mean.ts(), ut.run_var.ts(), ut.eps, ut.A.ts(), ut.B.ts(), ut.k) :
                    eg.batchNorm_leakyRelu(inplace, X, ut.run_mean.ts(), ut.run_var.ts(), ut.eps, ut.k));
            dX_mean.delete(); dX_var.delete();//intermediate variables are not needed
            return Y;
        }
        //</editor-fold>
    
        //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
        @Override
        protected Tensor __backward_no_affine__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                    eg.batchNorm_leakyRelu_deltaX_v1(grad_inplace, deltaY, ut.k, holdY(), 
                            ut.run_var.ts(), ut.eps):
                    eg.batchNorm_leakyRelu_deltaX_v2(grad_inplace, deltaY, ut.k, holdX(),
                            ut.run_mean.ts(), ut.run_var.ts(), ut.eps));
        }
    
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!ut.affine) return __backward_no_affine__(eg, deltaY, grad_inplace, backward_grads);//affine = false
            
            if(ut.A.need_grads() || ut.B.need_grads() || backward_grads) {
                Tensor[] grads = (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                        eg.batchNorm_leakyRelu_gradients_v1(grad_inplace, deltaY, ut.k, holdY(), 
                                ut.run_var.ts(), ut.eps, ut.A.ts(), ut.B.ts()) :
                        eg.batchNorm_leakyRelu_gradients_v2(grad_inplace, deltaY, ut.k, holdX(), 
                                ut.run_mean.ts(), ut.run_var.ts(), ut.eps, ut.A.ts(), ut.B.ts()));
            
                Tensor deltaX = grads[0];
                Tensor deltaA = grads[1];
                Tensor deltaB = grads[2];
                
                if(ut.A.need_grads()) ut.A.grads().add(deltaA); else deltaA.remote_delete();
                if(ut.B.need_grads()) ut.B.grads().add(deltaB); else deltaB.remote_delete();
             
                if(backward_grads) {  ut.A.ts().follow(deltaX); ut.B.ts().follow(deltaX); return deltaX; }
                else { deltaX.remote_delete(); return null; }
            }
        
            return null;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
