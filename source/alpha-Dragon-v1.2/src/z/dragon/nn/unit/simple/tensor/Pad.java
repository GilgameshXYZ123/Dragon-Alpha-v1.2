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
public class Pad extends SimpleInplaceFunction {
    private static final long serialVersionUID = 1L;
    
    protected int[] p0;//padding on the start
    protected int[] p1;//padding on the end
    
    public Pad(boolean inplace, int[] p0, int[] p1) {
        super(inplace);
        this.p0 = p0;
        this.p1 = p1;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int[] p0() { return p0; }
    public Pad p0(int... p) { this.p0 = p; return this; }
    
    public final int[] p1() { return p1; }
    public Pad p1(int... p) { this.p1 = p; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", p0 = ["); Vector.append(sb, p0); sb.append("]");
        sb.append(", p1 = ["); Vector.append(sb, p1); sb.append("] }");
    }
    
    @Override
    protected InlinePad create_unit_core() {
        return new InlinePad(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlinePad">
    public static class InlinePad extends SimpleInplaceInline<Pad> 
    {
        public InlinePad(Pad unit) { super(unit); }
        
        public final int[] p0() { return ut.p0; }
        public final int[] p1() { return ut.p1; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.pad(inplace, X, ut.p0, ut.p1);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads?
                    eg.trim(grad_inplace, deltaY, ut.p0, ut.p1):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
