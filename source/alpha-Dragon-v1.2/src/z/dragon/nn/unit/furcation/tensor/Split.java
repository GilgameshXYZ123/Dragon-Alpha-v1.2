/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.furcation.tensor;

import java.util.Arrays;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.furcation.FurcationCore;
import z.dragon.nn.unit.furcation.FurcateFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Split extends FurcateFunction {
    private static final long serialVersionUID = 1L;
    
    protected int dimIdx;
    protected int[] section;
    
    public Split(int dimIdx, int... section) {
        this.dimIdx = dimIdx;
        this.section = section;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int dimIdx() { return dimIdx; }
    public Split dimIdx(int dimIndex) { this.dimIdx = dimIndex; return this; }
    
    public final int[] section() { return section; }
    public Split section(int... section) { this.section = section; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { dimIdx = ").append(dimIdx);
        sb.append(", section = ").append(Arrays.toString(section)).append(" }");
    }
    
    @Override
    protected InlineSplit create_unit_core() {
        return new InlineSplit(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineSplit"> 
    public static class InlineSplit extends FurcationCore<Split>
    {
        public InlineSplit(Split unit) { super(unit); }

        public final int dimIdx() { return ut.dimIdx; }
        public final int[] section() { return ut.section; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor[] __forward__(Engine eg, Tensor X) {
            return eg.split(X, ut.dimIdx, ut.section);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor[] deltaY, 
                boolean grad_inplace, boolean[] grad_inplaces, 
                boolean backward_grads)
        {
            if(!backward_grads) return null;
        
            Tensor deltaX = eg.concat(ut.dimIdx, deltaY);
        
            if(grad_inplace) deltaX.dual(()-> { 
                for(int i=0; i<grad_inplaces.length; i++)
                    if(grad_inplaces[i]) deltaY[i].delete();
            });
        
            return deltaX;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
