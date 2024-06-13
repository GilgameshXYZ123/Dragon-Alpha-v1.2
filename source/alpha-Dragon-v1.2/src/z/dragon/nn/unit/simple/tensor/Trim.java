/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.tensor;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Trim extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 1L;
    
    protected int[] t0;//trimming on the start
    protected int[] t1;//trimming on the end
    
    public Trim(boolean inplace, int[] t0, int[] t1) {
        super(inplace);
        this.t0 = t0;
        this.t1 = t1;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int[] t0() { return t0; }
    public Trim t0(int... t) { this.t0 = t; return this; }
    
    public final int[] t1() { return t1; }
    public Trim t1(int... t) { this.t1 = t; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", t0 = ["); Vector.append(sb, t0); sb.append("]");
        sb.append(", t1 = ["); Vector.append(sb, t1); sb.append("] }");
    }
    
    @Override
    protected InlineTrim create_unit_core() {
        return new InlineTrim(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineTrim">
    public static class InlineTrim extends SimpleInplaceInline<Trim> 
    {
        public InlineTrim(Trim unit) { super(unit); }
        
        public final int[] t0() { return ut.t0; }
        public final int[] t1() { return ut.t1; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.trim(inplace, X, ut.t0, ut.t1);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads?
                    eg.pad(grad_inplace, deltaY, ut.t0, ut.t1):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
