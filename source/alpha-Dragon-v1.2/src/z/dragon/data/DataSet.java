/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import z.dragon.data.container.BatchIter;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.data.container.DataContainer;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;

/**
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public class DataSet<K, V>
{
    protected DataContainer<K, V> con;
    protected Transform<K[]> key_transform;
    protected Transform<V[]> value_transform;
    
    public DataSet(DataContainer<K, V> conta) { this.con = conta; }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public DataSet set_seed(long seed) { con.seed_seed(seed); return this; }
    
    public Transform<K[]>  input_transform() { return key_transform; }
    public DataSet<K, V> input_transform(Transform<K[]> input_transform) {
        if(input_transform == null) throw new NullPointerException("input_transform is null");
        this.key_transform = input_transform;
        return this;
    }
    
    public Transform<V[]>  label_transform() { return value_transform; }
    public DataSet<K, V> label_transform(Transform<V[]> label_transform) {
        if(label_transform == null) throw new NullPointerException("label_transform is null");
        this.value_transform = label_transform;
        return this;
    }
    
    public DataContainer<K, V> container() { return con; }
    public DataSet<K, V> container(DataContainer<K, V> conta) {
        if(conta == null) throw new NullPointerException("DataContainer is null");
        this.con = conta;
        return this;
    }
    
    public int size() { return con.size(); }
    public Class<K> input_class() { return con.input_class(); }
    public Class<V> label_class() { return con.label_class(); }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" { ");
        sb.append("size = ").append(size());
        sb.append(", input_class").append(input_class());
        sb.append(", label_class").append(label_class());
        sb.append("}");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(256);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    //<editor-fold defaultstate="collapsed" desc="operators: inner-code">
    protected static final ThreadFactory daemonThreadFactory = (Runnable r) -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    };
    protected final ExecutorService exec = Executors.newFixedThreadPool(2, daemonThreadFactory); 
    
    protected TensorPair create_tensor_pair(Engine eg, Pair kv) {
        K[] inputs = (K[]) kv.input;
        V[] labels = (V[]) kv.label;
        
        Future<Tensor> finput = exec.submit(() -> { return key_transform.transform(eg, inputs); });
        Future<Tensor> flabel = exec.submit(() -> { return value_transform.transform(eg, labels); });

        Tensor input, label;
        try { input = finput.get(); label = flabel.get(); }
        catch(InterruptedException | ExecutionException e) { 
            throw new RuntimeException(e); 
        }
        return new TensorPair(input, label);
    }
    //</editor-fold>
    
    public DataSet<K, V> shuffle_sort() { con.shuffle_sort(); return this; }
    public DataSet<K, V> shuffle_swap(double percent) { con.shuffle_swap(percent); return this; }
    public DataSet<K, V> shuffle_swap() { con.shuffle_swap(); return this; }
    
    public TensorPair get(Engine eg) { return create_tensor_pair(eg, con.get(1)); }
    public TensorPair get(Engine eg, int batch) { return create_tensor_pair(eg, con.get(batch)); }
    public void clear() { con.clear(); }
    
    public DataSet<K, V>[] split(float percent) { return split((int)(size() * percent)); }
    public DataSet<K, V>[] split(int sub_size) {
        DataContainer<K, V>[] contas = con.split(sub_size);
        DataSet<K, V> first = new DataSet<>(contas[0]);
        DataSet<K, V> last  = new DataSet<>(contas[1]);
        return new DataSet[]{ first, last };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="batch_iterator">
    protected class BIter implements TensorIter
    {
        private final BatchIter iter;
        
        public BIter(BatchIter iter) { this.iter = iter; }
        
        @Override public TensorIter shuffle_sort() { DataSet.this.shuffle_sort(); return this; }
        @Override public TensorIter shuffle_swap() { DataSet.this.shuffle_swap(); return this; }
        @Override public TensorIter shuffle_swap(double percent) { DataSet.this.shuffle_swap(percent); return this; }
       
        @Override public TensorIter reset() { iter.reset(); return this; }
        @Override public boolean hasNext() { return iter.hasNext(); }
        @Override
        public TensorPair next(Engine eg, int batch, boolean random) {
            return create_tensor_pair(eg, iter.next(batch, random));
        }
    }
    //</editor-fold>
    public TensorIter batch_iter() {  return new BIter(con.batch_iter());  }
    
    public BufferedTensorIter buffered_iter(Engine eg, int batch_size) { 
        return new BufferedTensorIter(batch_iter(), eg, batch_size);
    }
}
