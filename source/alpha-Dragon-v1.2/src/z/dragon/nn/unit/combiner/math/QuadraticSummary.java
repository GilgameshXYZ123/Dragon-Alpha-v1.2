/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.combiner.math;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.combiner.CombinerCore;
import z.dragon.nn.unit.combiner.CombinerFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class QuadraticSummary extends CombinerFunction {
    private static final long serialVersionUID = 1L;
    
    protected float alpha;
    protected float beta;
    protected float gamma;
    
    public QuadraticSummary(float alpha, float beta, float gamma) {
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final float alpha() { return alpha; }
    public QuadraticSummary alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float beta() { return beta; }
    public QuadraticSummary beta(float beta) { this.beta = beta; return this; }
    
    public final float gamma() { return gamma; }
    public QuadraticSummary gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append("{ alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    
    @Override
    protected InlineQuadraticSummary create_unit_core() {
        return new InlineQuadraticSummary(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineQuadraticSummary"> 
    public static class InlineQuadraticSummary extends CombinerCore<QuadraticSummary>
    {
        public InlineQuadraticSummary(QuadraticSummary unit) { super(unit); }
        
        public final float alpha() { return ut.alpha; }
        public final float beta() { return ut.beta; }
        public final float gamma() { return ut.gamma; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor[] X) {
            return eg.quadratic_sum(false, ut.alpha, ut.beta, ut.gamma, X);
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
                int input_tensor_num, boolean grad_inplace, 
                boolean backward_grads, boolean[] last_need_grads) 
        {
            if(!backward_grads) return null;//grads_num = 0;
        
            Tensor[] deltaX = new Tensor[input_tensor_num];
            int[] backward_grads_index = backward_grads_index(last_need_grads);
            int grads_num = backward_grads_index.length;
        
            float alpha2 = 2.0f * ut.alpha;
            if(grads_num == 1) {
                int idx = backward_grads_index[0];
                deltaX[idx] = eg.quadratic2(grad_inplace, deltaY, holdX(idx), 
                        0.0f, alpha2, 0, ut.beta, 0, 0);
                return deltaX;
            }
        
            for(int i=0; i<grads_num; i++) {
                int idx = backward_grads_index[i];
                deltaX[idx] = eg.quadratic2(false, deltaY, holdX(idx), 
                        0.0f, alpha2, 0, 
                        ut.beta, 0, 0);
            }
        
            if(grad_inplace) {//when deltaX[] are found, deltaY is not needed
                CountGc gc = new CountGc(grads_num, deltaY);
                for (Tensor grad : deltaX) grad.dual(()-> { gc.countDown(); });
            }
            
            return deltaX;
        }
        //</editor-fold>
    }
    //</editor-fold>
}