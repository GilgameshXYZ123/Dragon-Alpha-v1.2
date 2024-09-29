/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data.container;

import java.util.Map;
import java.util.Random;
import z.dragon.data.Pair;

/**
 *
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public interface DataContainer<K, V> {
    public int size();
    default boolean isEmpty() { return size() == 0; }
    
    public DataContainer<K, V>[] split(int sub_size);
    default DataContainer<K, V>[] split(float percent) { return split((int)(size() * percent)); }
   
    //for (cls : classes): sub.cls.num = cls.num * percent
    public DataContainer<K, V>[] class_split(float percent);//split each class
    
     public Map<V, Integer> class_sample_num();
    public Class<K> input_class();
    public Class<V> label_class();

    public Random random();
    default void seed_seed(long seed) { random().setSeed(seed); }

    public static double default_rand_swap_percent = 0.25;
    public DataContainer<K, V> shuffle_sort();
    public DataContainer<K, V> shuffle_swap(double percent);
    default DataContainer<K, V> shuffle_swap() { return shuffle_swap(default_rand_swap_percent); }
    
    public void add(K key, V value);
    public void add(K[] keys, V[] values);
    
    public Pair<K, V> get();
    public Pair<K[], V[]> get(int batch);
     
    public void clear();
    
    public BatchIter<K[], V[]> batch_iter();
}
