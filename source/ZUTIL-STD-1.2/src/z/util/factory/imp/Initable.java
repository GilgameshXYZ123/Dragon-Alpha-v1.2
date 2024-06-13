/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.factory.imp;

/**
 *
 * @author dell
 * @param <T>
 */
public interface Initable<T>
{
    public void init(T columns) throws Exception;
}
