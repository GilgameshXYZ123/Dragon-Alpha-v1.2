/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data.container;

import z.dragon.data.Pair;

/**
 *
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public interface BatchIter<K, V> {
    public BatchIter<K, V> shuffle_swap(float percent);
    public BatchIter<K, V> shuffle_sort();
    
    public BatchIter<K, V> reset();
    
    public boolean hasNext();
    
    default Pair<K, V> next(int batch) { return next(batch, false); }
    public Pair<K, V> next(int batch, boolean random);
}
