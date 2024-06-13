/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool.imp;

/**
 * To contain the resource of a pool, any DataSturcture meets the need 
 * can implement this Interface.
 * @author dell
 * @param <T>
 */
public interface PoolBuffer<T> extends Iterable<T>
{
    public boolean isEmpty();
    public int number();
    public int size();
    public T remove();
    public boolean add(T e);
    public void clear();
}
