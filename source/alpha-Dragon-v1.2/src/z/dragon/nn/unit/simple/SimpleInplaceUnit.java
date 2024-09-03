/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple;

/**
 *
 * @author Gilgamesh
 */
public abstract class SimpleInplaceUnit extends SimpleUnit {
    private static final long serialVersionUID = 1L;
    
    boolean inplace;
     
    protected SimpleInplaceUnit(boolean inplace) { this.inplace = inplace; }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final boolean inplace() { return inplace; }
    public SimpleInplaceUnit inplace(boolean inplace) { this.inplace = inplace; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace()).append(" }");
    }
    //</editor-fold>
}
