/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.module;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.UnitCore;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreModule<T extends Module> extends UnitCore<T>
{
    private final CoreGraphHead head;
    private final CoreGraphTail tail;
    
    transient private Tensor[] X, deltaX;//input, input.gradient
    transient private Tensor[] Y, deltaY;//output, output.gradient

    public CoreModule(T module) { 
        super(module); 
        head = new CoreGraphHead(module, this);
        tail = new CoreGraphTail(module, this);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    public final Tensor[] X() { return X; }
    public final Tensor[] Y() { return Y; }
    
    public final Tensor[] deltaX() { return deltaX; }
    public final Tensor[] deltaY() { return deltaY; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public Collection<UnitCore<?>> next() { return tail.nexts; }
    
    @Override public void variables(Tensor.TensorSet set) { head.variables(set); tail.variables(set); }
    @Override public void gc() { head.gc(); tail.gc();}
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    @Override
    public Tensor[] forward(Tensor... input) {
        input = head.forward(X = input);//this.starts = head.next
        return Y = tail.forward(ut.__forward__(input));//this.next = tail.next;
    }

    @Override
    protected void traceBack(UnitCore<?> next, int out_index, int next_in_index) {
        tail.traceBack(next, out_index, next_in_index);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override public synchronized Tensor[] collectGradientFromNext() { return tail.collectGradientFromNext();  }
    @Override public synchronized Tensor gradient(int index) {  return head.gradient(index); }

    transient private final HashSet<UnitCore> visited = new HashSet<>(4);
    
    /**
     * <pre>.
     * if Unit instance of module
     * (1) for(Unit next : module.nexts == module.tail.next) backward(next, backed);
     * (2) module.collectGradientFromNext() = module.tail.collectGradientFromNext()
     *  As: module.nexts = tail.nexts, 
     * (3) module.backward(gradient): {@code 
     *      module.deltaX <- head.deltaX <- head.collectFromNext()
     *          <- head.next.backward(head.next.collectFromNext()) - ....
     *          <- tail.last.backward(tail.last.collectFromNext()) <- tail.backward(gradient) }
     * [1] when: module.hasNext -> tail.hasNext
     *  module.collectFromNext is called: -> module.tail.collectGradientFromNext is called
     * [2] when: !module.hasNext -> tail.hasNoNext:  
     *  so module, module.tail is the end of global compute graph, 
     *  just set: module.deltaY = tail.deltaX = tail.deltaY
     * @param node 
     */
    private void backward(UnitCore<?> node) {
        if(visited.contains(node)) return;//tail is visited, head will be visited
      
        for(UnitCore<?> next : node.next()) backward(next);//go to the leaf node in the compute graph
        node.backward(node.collectGradientFromNext());//ends.collect gradients from tail
        
        visited.add(node);
    }
    
    @Override //the gradients must be alinged with input, no need to reorder
    public synchronized Tensor[] backward(Tensor... gradient) {
        //the end node of graph, tail.deltaY = this.deltaY = gradient
        tail.backward(deltaY = gradient);//assign gradient to tail.gradient
        visited.add(tail);
        
        for(UnitCore<?> start : head.nexts) backward(start);//head.next = graph.start
        visited.clear(); 
       
        //head computes the gradient for input, based on sub arcs of sub graph of Module
        deltaX = head.collectGradientFromNext();
        
        //final process---------------------------------------------------------
        if(deltaX != null) //collect gradient for deltaX
            for(int i=0; i<X.length; i++)  if(X[i].need_grad()) X[i].grad(deltaX[i]);
        
        return deltaX;
    }
    //</editor-fold>
}
