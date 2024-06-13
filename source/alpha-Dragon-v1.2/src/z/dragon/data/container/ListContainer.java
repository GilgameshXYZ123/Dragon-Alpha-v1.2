/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data.container;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import z.dragon.data.Pair;
import z.util.math.vector.Vector;

/**
 * @author Gilgamesh
 * @param <K>
 * @param <V> 
 */
 @SuppressWarnings(value = "unchecked")
public class ListContainer<K, V> extends AbstractContainer<K, V>
{
    protected final ArrayList<K> karr;
    protected final ArrayList<V> varr;
   
    public ListContainer(Class<K> input_clazz, Class<V> label_clazz, int init_capacity) {
       super(input_clazz, label_clazz);
        karr = new ArrayList<>(init_capacity);
        varr = new ArrayList<>(init_capacity);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public int size() { return karr.size(); }
    public List<K> inputs() { return karr;}
    public List<V> labels() { return varr; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators: write">
    @Override
    public ListContainer<K, V> shuffle_sort() {
        start_read();
        int size = karr.size();
        int[] key = Vector.random_int_vector(size);
        Pair[] val = new Pair[size];
        for(int i=0; i<size; i++) val[i] = new Pair(karr.get(i), varr.get(i));
        finish_read();
        
        Vector.sort(key, val);
        
        start_write();
        for(int i=0; i<size; i++) {
            karr.set(i, (K) val[i].input);
            varr.set(i, (V) val[i].label);
        }
        finish_write();
        return this;
    }
    
    protected final void __random_swap(int n) {
        int size = karr.size(), len = (n + 1) >> 1;
        for(int i=0; i<len; i++) {
            int index1 = exr.nextInt(0, size - 1);
            int index2 = exr.nextInt(0, size - 1);
            K k = karr.get(index1); karr.set(index1, karr.get(index2)); karr.set(index2, k);
            V v = varr.get(index1); varr.set(index1, varr.get(index2)); varr.set(index2, v);
        }
    }
    
    @Override
    public ListContainer<K, V> shuffle_swap(double percent) {
        if(percent <= 0.0f) throw new IllegalArgumentException(String.format(
                "percent { got %f } must > 0", percent));
        if(percent >= Integer.MAX_VALUE) throw new IllegalArgumentException(String.format(
                "percent { got %f } must < Integer.MAX { got %d }", percent, Integer.MAX_VALUE));
        double op = (int) percent, rp = percent - op;
        
        start_write();
        int size = karr.size(), rsize = (int) Math.ceil(rp * size);
        for(int i=0; i<op; i++) __random_swap(size);//integeral part
        __random_swap(rsize);//fractional part;
        finish_write();
        return this;
    }
    
      public void ensureCapacity(int capacity) {
        if(capacity <= 0) throw new IllegalArgumentException(String.format(
                "capacity { got %d } must > 0", capacity));
        start_write();
        karr.ensureCapacity(capacity);
        varr.ensureCapacity(capacity);
        finish_write();
    }
    
    @Override
    public void add(K input, V label) {
        if(input == null || label == null) return;
        start_write();
        karr.add(input); 
        varr.add(label);
        finish_write();
    }
    
    @Override
    public void add(K[] inputs, V[] labels) {
        if(inputs == null || labels == null) return;
        if(inputs.length != labels.length) throw new IllegalArgumentException(String.format(
                "inputs.length[%d] != labels.length[%d]", inputs.length, labels.length));
        
        start_write();
        karr.ensureCapacity(karr.size() + inputs.length);
        varr.ensureCapacity(varr.size() + labels.length);
        for(int i=0; i<inputs.length; i++) {
            if(inputs[i] == null || labels[i] == null) continue;
            karr.add(inputs[i]);
            varr.add(labels[i]);
        }
        finish_write();
    }
    
    @Override
    public void clear() {
        start_write();
        karr.clear(); 
        varr.clear();
        finish_write();
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="operators: read">
    @Override
    public Pair<K, V> get() {
        start_read();
        if(karr.isEmpty() || varr.isEmpty()) throw new NullPointerException("container is empty");
        int index = exr.nextInt(0,  karr.size() - 1);
        K k = karr.get(index);
        V v = varr.get(index);
        finish_read();
        
        return new Pair<>(k, v);
    }
     
    @Override
    public Pair<K[], V[]> get(int batch) {
        if(batch <= 0) throw new IllegalArgumentException(String.format(
                "batch { got %d }must > 0", batch));
        
        K[] ks = (K[]) Array.newInstance(kclazz, batch);
        V[] vs = (V[]) Array.newInstance(vclazz, batch);
        
        start_read();
        if(karr.isEmpty() || varr.isEmpty()) throw new NullPointerException("container is empty");
        for(int i=0, size = karr.size(); i<batch; i++) {
            int index = exr.nextInt(0, size - 1);
            ks[i] = karr.get(index);
            vs[i] = varr.get(index);
        }
        finish_read();
      
        return new Pair<>(ks, vs);
    }
    
    @Override
    public ListContainer<K, V>[] split(int sub_size) {
        if(sub_size <= 0) throw new IllegalArgumentException(String.format("sub_size (%d) <= 0", sub_size));
        if(sub_size >= size()) throw new IllegalArgumentException(String.format("sub_size (%d) >= size", sub_size));
        
        start_read();
        int size = size();
        ListContainer<K, V> first = new ListContainer(this.kclazz, this.vclazz, sub_size);
        ListContainer<K, V> last  = new ListContainer(this.kclazz, this.vclazz, size - sub_size);
        
        for(int i=0; i<sub_size; i++)   first.add(karr.get(i), varr.get(i));//[0:sub_size - 1]
        for(int i=sub_size; i<size; i++) last.add(karr.get(i), varr.get(i));//[sub_size : size - 1]
        finish_read();
        
        return new ListContainer[]{ first, last};
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="class BatchIterater">
    class BIter implements BatchIter<K[], V[]>
    {
        private int index = 0;

        @Override
        public final BatchIter<K[], V[]> shuffle_swap(float percent) {
            ListContainer.this.shuffle_swap(percent);
            return this;
        }

        @Override
        public final BatchIter<K[], V[]> shuffle_sort() {
            ListContainer.this.shuffle_sort();
            return this;
        }

        @Override public final BatchIter<K[], V[]> reset() { index = 0; return this; }
        @Override public final boolean hasNext() { return index < karr.size(); }
        
        //<editor-fold defaultstate="collapsed" desc="inner-code: next(batch)">
        final void next_inorder(K[] ks, V[] vs, int batch) {
            for(int i=0, size = karr.size(); i<batch; i++) {
                int mod_index = index % size;
                ks[i] = karr.get(mod_index);
                vs[i] = varr.get(mod_index);
                index++;
            }
        }
        
        final void next_random(K[] ks, V[] vs, int batch) {
             for(int i=0, size = karr.size(); i<batch; i++) {
                int mod_index = exr.nextInt(0, size - 1);
                ks[i] = karr.get(mod_index);
                vs[i] = varr.get(mod_index);
                index++;
            }
        }
        //</editor-fold>
        @Override
        public final Pair<K[], V[]> next(int batch, boolean random)  {
            if(batch <= 0) throw new IllegalArgumentException(String.format(
                    "batch { got %d } must > 0", batch));
            
            K[] ks = (K[]) Array.newInstance(kclazz, batch);
            V[] vs = (V[]) Array.newInstance(vclazz, batch);
            
            start_read();
            if(karr.isEmpty() || varr.isEmpty()) throw new NullPointerException("container is empty");
            if(random) next_random(ks, vs, batch); 
            else next_inorder(ks, vs, batch);
            finish_read();
            
            return new Pair<>(ks, vs);
        }
    }
    //</editor-fold>
    @Override public BatchIter<K[], V[]> batch_iter() { return new BIter(); }
}
