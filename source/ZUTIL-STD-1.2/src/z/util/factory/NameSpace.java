/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.factory;

import java.util.Collection;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import z.util.ds.tree.ZTreeMap;
import z.util.xml.exception.UnknownTagException;
import z.util.xml.XML;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 */
public class NameSpace
{
    String name;
    ZTreeMap<String, Resource> res=new ZTreeMap<>();
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    public String getName() 
    {
        return name;
    }
    public void setName(String name) 
    {
        this.name = name;
    }
    public void append(StringBuilder sb)
    {
        sb.append("NameSpace = ").append(name);
        sb.append("\nResources={");
        res.forEach((String key, Resource value)->{sb.append('\n').append(value);});
        sb.append("\n}");
    }
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        this.append(sb);
        return sb.toString();
    }
    public String toDetailString()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("NameSpace = ").append(name);
        sb.append("\nResources={");
        res.forEach((String key, Resource value)->{sb.append('\n').append(value.toDetailString());});
        sb.append("\n}");
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Operators">
    public Resource put(String name, Resource value)
    {
        return res.put(name, value);
    }
    public void putAll(NameSpace another)
    {
        res.putAll(another.res);
    }
    public Resource get(String name)
    {
        Resource rs=res.get(name);
        if(rs==null) throw new NullPointerException("No such Resource: "+name);
        return rs;
    }
    public Resource remove(String name)
    {
        return res.remove(name);
    }
    public Collection<Resource> values()
    {
        return res.values();
    }
    public void merge(NameSpace another)
    {
        if(!this.name.equals(another.name)) 
            throw new RuntimeException("Namespace.name are not the same: "+name+" != "+another.name);
        this.res.putAll(another.res);
        another.res.clear();
    }
    //</editor-fold>
    @Passed
    public static NameSpace valueOf(Node namespace) throws Exception
    {
        NameSpace ns=new NameSpace();
        
        NamedNodeMap attr=namespace.getAttributes();
        ns.name=XML.getAttributeNoNull(attr, "name");
        String pack=XML.getAttributeOrDefault(attr, "package", "");
        if(!pack.endsWith(".")) pack=pack.concat(".");
        
        NodeList child=namespace.getChildNodes();
        Node cur=null;
        String name=null;
        for(int i=0,len=child.getLength();i<len;i++)
        {
            cur=child.item(i);
            if(cur.getNodeType()!=Node.ELEMENT_NODE) continue;
            name=cur.getNodeName();
            if(!name.equals("resource")) throw new UnknownTagException(name);
            Resource rs=Resource.valueOf(cur, pack);
            ns.res.put(rs.name, rs);
        }
        return ns;
    }
}
