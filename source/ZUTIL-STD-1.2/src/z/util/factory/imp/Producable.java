/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.factory.imp;

/**
 *
 * @author dell
 * @param <V>
 */
public interface Producable<V>
{
    public <T extends V> T instance(String namespace,String name) throws Exception;
}