/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool.imp;

import java.lang.reflect.Array;
import z.util.pool.exception.CheckException;
import z.util.pool.exception.CreateException;
import z.util.pool.exception.DestroyException;

/**
 * <pre>
 * the kernel of the pool.
 * It includes such basic functions, such as:
 * (1) create: create an unit resource.
 * Ex: create a Connection to a specific database.
 * (2) check:  check the reosurce is useable or not.
 * Ex: check a Connection is closed or not, or the resource if null or not.
 * (3) destroy: destroy a unit of resource.
 * Ex: close an connection to DataBase.
 * </pre>
 * @author dell
 * @param <T>
 */
public interface PoolManager<T> 
{
    public T create() throws CreateException;
    public boolean check(T resource) throws CheckException;
    public void destroy(T resource) throws DestroyException;
    
    default T[] create(int size) 
    {
        T[] res=(T[]) Array.newInstance(this.getClass().getComponentType(), size);
        for(int i=0;i<size;i++) res[i]=this.create();
        return res;
    }
}
