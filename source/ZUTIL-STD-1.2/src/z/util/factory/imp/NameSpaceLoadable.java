/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.factory.imp;

/**
 *
 * @author dell
 * @param <A>
 * @param <B>
 */
public interface NameSpaceLoadable<A, B>
{
    public void load(A src) throws Exception;
    public void removeNameSpace(B src) throws Exception;
    public void clearNameSpace() throws Exception;
}
