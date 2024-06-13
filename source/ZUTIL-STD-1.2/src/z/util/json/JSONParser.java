/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.json;

/**
 *
 * @author dell
 */
public abstract class JSONParser
{
    public abstract void append(StringBuilder sb, Object obj) throws Exception;
}
