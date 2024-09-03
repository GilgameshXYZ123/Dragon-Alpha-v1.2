/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core;

/**
 * 
 * @author Gilgamesh
 */
public final class Trace {
    private final UnitCore<?> last;
    private final int out_index;//Tensor = out.Y[pre_out_index]
    private final boolean need_grads;

    public Trace(UnitCore<?> last, int pre_out_index, boolean need_grads) {
        this.last = last;
        this.out_index = pre_out_index;
        this.need_grads = need_grads;
    }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public final UnitCore<?> last_node() { return last; }
    public final int last_out_index() { return out_index; }
    public final boolean need_grads() { return need_grads; }

    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append("{ ");
        sb.append("need_grads = ").append(need_grads);
        sb.append(", last_node = ").append(last.name());
        sb.append(", last_out_index = ").append(out_index);
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(64);
        this.append(sb);
        return sb.toString();
    }
    
    public final void callback(UnitCore<?> next, int next_in_index) {
        last.traceBack(next, out_index, next_in_index);
    }
    //</editor-fold>
}
