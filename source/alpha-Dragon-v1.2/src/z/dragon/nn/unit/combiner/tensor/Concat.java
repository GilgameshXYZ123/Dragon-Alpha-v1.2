/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.combiner.tensor;

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
public class Concat extends CombinerFunction {
    private static final long serialVersionUID = 1L;
    
    protected int dimIdx;
    
    public Concat(int dimIdx) { this.dimIdx = dimIdx; }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int dimIdx() { return dimIdx; }
    public Concat dimIdx(int dimIndex) { this.dimIdx = dimIndex; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { dimIdx = ").append(dimIdx).append(" }");
    }
    
    @Override
    protected InlineLConcat create_unit_core() {
        return new  InlineLConcat(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLinearMean"> 
    public static class InlineLConcat extends CombinerCore<Concat>
    {
        public InlineLConcat(Concat unit) { super(unit); }

        transient private int[] section;
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor[] X) {
            Tensor Y = eg.concat(ut.dimIdx, X);
            section = new int[X.length];
            for(int i=0; i<X.length; i++) section[i] = X[i].dim(ut.dimIdx);
            return Y;
        }

        @Override
        protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
               int input_tensor_num, boolean grad_inplace, 
               boolean backward_grads, boolean[] last_need_grads)
        {
            if(!backward_grads) return null;//grads_num = 0;
            
            Tensor[] deltaX = eg.split(deltaY, last_need_grads, ut.dimIdx, section);
            
            if(grad_inplace) {//when deltaX[] are found, deltaY is not needed
                int grad_num = last_need_grads.length;
                int[] backward_grads_index = backward_grads_index(last_need_grads);
                CountGc gc = new CountGc(grad_num, deltaY);
                for(int i=0; i< grad_num; i++) {
                    int idx = backward_grads_index[i];
                    deltaX[idx].dual(()-> { gc.countDown(); });
                }
            }
        
            return deltaX;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
