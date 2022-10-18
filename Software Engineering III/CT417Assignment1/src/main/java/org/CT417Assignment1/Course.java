package org.CT417Assignment1;

import org.joda.time.DateTime;

import java.util.ArrayList;

public class Course
{
    private String name;
    private ArrayList<Module> modules;
    private ArrayList<Student> students;
    private DateTime startDate;
    private DateTime endDate;

    public Course(String name, ArrayList<Module> modules, ArrayList<Student> students, DateTime startDate, DateTime endDate)
    {
        this.name = name;
        this.modules = modules;
        this.students = students;
        this.startDate = startDate;
        this.endDate = endDate;
    }

    public String GetName()
    {
        return this.name;
    }

    public ArrayList<Module> GetModules()
    {
        return this.modules;
    }

    public ArrayList<Student> GetStudents()
    {
        return this.students;
    }

    public DateTime GetStartDate()
    {
        return startDate;
    }

    public DateTime GetEndDate()
    {
        return endDate;
    }

}
