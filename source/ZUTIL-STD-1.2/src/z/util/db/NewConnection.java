/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.db;

import java.sql.Connection;

/**
 *
 * @author dell
 * @param <T>
 */
public interface NewConnection<T>
{
    public T newConnection() throws Exception;
    public void releaseConnection(T con) throws Exception;
}
