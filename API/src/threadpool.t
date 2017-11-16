tp = {}

local C = terralib.includecstring [[
#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
]]
local I = require('ittnotify')

c = require('config')
numthreads = c.numthreads

-- local DEBUG_MUTEX = 1
local DEBUG_MUTEX = 0
local debm = macro(function(apicall)
if DEBUG_MUTEX == 1 then
  return quote apicall end
else
  return quote end
end
end)

-- checked pthread call
pth = {}
pth[16] = 'EBUSY'
-- for k,v in pairs(C) do print(k,v) end
-- local asdf = C.ESRCH

ptcode = global(int, 0, "ptcode")
local pt = macro(function(apicall)
  local apicallstr = tostring(apicall)
  return quote 
    var str = [apicallstr]
    ptcode = apicall
    if ptcode ~= 0 then
      C.printf('ERROR IN PTHREADS CALL, errorcode = %d\n', ptcode)
      C.printf('        in call: %s\n', str)
    end
  end
end)

-------------------------------- GLOBALS START
local theThreads = global(C.pthread_t[numthreads], nil, 'theThreads')
tp.theThreads = theThreads


-- WAIT FOR WORK - SEMAPHORE
-- Summary: Synch mechanism used by main-thread to tell worker-threads that
--          work is available for them.
-- init by: initThreads (on main thread)
-- destroyed by: joinThreads (on main thread)
-- used by: (together, wait:worker(individual), signal:main)
-- *  waitForWork(): wait until work is available for this thread.
-- *  TaskQueue_t:set(): signal worker thread that work is available.
-- NOTE: no need to export these globals because they are only used within
--       this file. We still define them as global because even within this file
--       we need access from many points.
local thread_busy_mutex = global(C.pthread_mutex_t[numthreads], nil, "thread_busy_mutex")
local work_available_cv = global(C.pthread_cond_t[numthreads], nil, "work_available_cv")

-- TODO this is awful programming style, change it!
-- TODO implement singleton pattern for all synch mechanisms
local struct WaitForWorkSemaphore {
  -- 'members' are the global from above
}
local theWaitForWorkSemaphore = global(WaitForWorkSemaphore, nil, 'theWaitForWorkSemaphore')
terra WaitForWorkSemaphore:init()
  for k = 0,numthreads do                                                       
    debm( C.printf('initThreads(): value of thread_busy_mutex[%d]\
                    before init is %d\n', k, thread_busy_mutex[k]) )
    pt( C.pthread_mutex_init(&thread_busy_mutex[k], nil))
    debm( C.printf('initThreads(): value of thread_busy_mutex[%d]\
                    after init is %d\n', k, thread_busy_mutex[k]) )
    pt( C.pthread_cond_init(&work_available_cv[k], nil) )
  end                                                                           
end
terra WaitForWorkSemaphore:destroy()
  for k = 0,numthreads do                                                       
    pt( C.pthread_mutex_destroy(&thread_busy_mutex[k]) )
    pt( C.pthread_cond_destroy(&work_available_cv[k]))
  end                                                                           
end
terra WaitForWorkSemaphore:wait(tid : int)
    pt( C.pthread_cond_wait(&work_available_cv[tid],
        &thread_busy_mutex[tid]))
end

-- The following two are necessary because we need to lock a condition-var's (cv)
-- mutex before the first use of the cv and unlock it after the last use. It's
-- a bit of an ugly hack, maybe this can be refactored at a later point in time.
--
-- We need the tid arg because each thread has it's own cv and mutex
terra WaitForWorkSemaphore:initialLock(tid: int)
  pt( C.pthread_mutex_lock(&thread_busy_mutex[tid]) )
end
terra WaitForWorkSemaphore:finalUnlock(tid : int)
  pt( C.pthread_mutex_unlock(&thread_busy_mutex[tid]) )
end

terra WaitForWorkSemaphore:signal(receiver_tid : int)
  var domain = I.__itt_domain_create("Main.Domain")
  var name_signal = 
     I.__itt_string_handle_create('TaskQueue_t:set(): sending work signal')
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_signal)

  pt( C.pthread_mutex_lock(&thread_busy_mutex[receiver_tid]))

  debm( C.printf('TaskQueue_t:set(): signaling that work is\
                  available for thread %d\n', receiver_tid) )

  pt( C.pthread_cond_signal(&work_available_cv[receiver_tid]))

  I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_signal)

  debm( C.printf('TaskQueue_t:set(): starting to unlock\
                  thread_busy_mutex[%d]\n', receiver_tid) )
  pt( C.pthread_mutex_unlock(&thread_busy_mutex[receiver_tid]))
end


-- WORKLOAD FINISHED - BARRIER
-- Summary: Next two pairs (i.e. 4 globals) form a single entity. Synch mechanism
--          to ensure that all worker threads have finished their portion of
--          work before main-thread launches next kernel.
-- init by: initThreads (on main thread)
-- destroyed by: joinThreads (on main thread)
-- used by: (together, wait:main, signal:worker)
-- *  GPULauncher(): reset numkernels_finished to zero
-- *  GPULauncher(): wait for kernel_finished_cv
-- *  taskfunc(): when portion of work is complete, increase numkernel_finished
--                and signal (kernel_finished_cv) main thread that it may
--                continue
local numkernels_finished_mutex = global(C.pthread_mutex_t, nil, "numkernels_finished_mutex")
local numkernels_finished = global(int, 0, "numkernels_finished")                     
--
local kernel_running_mutex = global(C.pthread_mutex_t, nil, "kernel_running_mutex")   
local kernel_finished_cv = global(C.pthread_cond_t, nil, "kernel_finished_cv")        
tp.numkernels_finished_mutex = numkernels_finished_mutex
tp.numkernels_finished = numkernels_finished
tp.kernel_running_mutex = kernel_running_mutex
tp.kernel_finished_cv = kernel_finished_cv
-- TODO this is awful programming style, change it!
-- TODO implement singleton pattern for all synch mechanisms
local struct KernelFinishedByAllThreadsBarrier {
  -- 'members' are the global from above
}
local theKernelFinishedByAllThreadsBarrier = global(KernelFinishedByAllThreadsBarrier, nil, 'theKernelFinishedByAllThreadsBarrier')
tp.theKernelFinishedByAllThreadsBarrier = theKernelFinishedByAllThreadsBarrier
terra KernelFinishedByAllThreadsBarrier:init()
  pt( C.pthread_mutex_init(&numkernels_finished_mutex, nil) )
  pt( C.pthread_mutex_init(&kernel_running_mutex, nil) )
  pt( C.pthread_cond_init(&kernel_finished_cv, nil) )
  numkernels_finished = 0                                                     
end
terra KernelFinishedByAllThreadsBarrier:destroy()
  pt( C.pthread_mutex_destroy(&numkernels_finished_mutex))
  pt( C.pthread_mutex_destroy(&kernel_running_mutex) )
  pt( C.pthread_cond_destroy(&kernel_finished_cv) )
end
terra KernelFinishedByAllThreadsBarrier:wait()
  pt( C.pthread_cond_wait(&tp.kernel_finished_cv, &tp.kernel_running_mutex))

  -- reset numkernels_finished to zero
  debm( C.printf('GPULauncher(): locking numkernels_finished_mutex\n') )
  pt( C.pthread_mutex_lock(&numkernels_finished_mutex))
  debm( C.printf('GPULauncher(): setting numkernels_finished to zero\n') )
  numkernels_finished = 0                                                     
  debm( C.printf('GPULauncher(): unlocking numkernels_finished_mutex\n') )
  pt( C.pthread_mutex_unlock(&numkernels_finished_mutex))
end

-- The following two are necessary because we need to lock a condition-var's (cv)
-- mutex before the first use of the cv and unlock it after the last use. It's
-- a bit of an ugly hack, maybe this can be refactored at a later point in time.
--
-- TODO at the moment we are calling initialLock() inside GPULauncher *every*
-- time before giving work to the workers and *every* time after all threads
-- are done. So between to calls to GPULaucher, we are unlocking at the end
-- of GPULauncher1 and locking again at the start of GPULauncher2.
-- This seems unnecessary. It should be possible to simply put initialLock
-- into init() and finalLock into destroy() (same for the other barrier above)
--
-- Note: Have look at 'Note' in the Next barrier why this is not strictly
-- a barrier. However, THIS barrier resets after every use of wait()
--
-- This function needs to be called before *every* wait() call *before* giving
-- work to the worker-threads.
terra KernelFinishedByAllThreadsBarrier:initialLock()
  pt( C.pthread_mutex_lock(&tp.kernel_running_mutex))
end
-- This function should be called directly after every wait-call
terra KernelFinishedByAllThreadsBarrier:finalUnlock()
  pt( C.pthread_mutex_unlock(&tp.kernel_running_mutex))
end

terra KernelFinishedByAllThreadsBarrier:signal()
  var domain = I.__itt_domain_create("Main.Domain")
  -- signal 'numkernels_finished' to say that this thread is finished.
  -- If this is the last thread to finish.....(see below)
  pt( C.pthread_mutex_lock(&tp.numkernels_finished_mutex) )
  debm( C.printf("taskfun(): increasing numkernels_finished counter\n") )
  tp.numkernels_finished = tp.numkernels_finished + 1                               
   
  debm( C.printf("taskfun(): checking if all threads are done\n") )
  -- ... (cont.) then signal main thread that it is allowed to continue
  -- and return from here into the worker-thread's infinite wait-loop
  if tp.numkernels_finished == numthreads then                                   

    var name = I.__itt_string_handle_create('taskfunc(): sending kernel_finished signal')
    I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
    pt( C.pthread_mutex_lock(&tp.kernel_running_mutex) )
    pt( C.pthread_cond_signal(&tp.kernel_finished_cv) )
    pt( C.pthread_mutex_unlock(&tp.kernel_running_mutex) )
    I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)
  end                                                                         
  pt( C.pthread_mutex_unlock(&tp.numkernels_finished_mutex) )
end



-- WAIT FOR WORKER THREADS TO LIVE - BARRIER
-- Summary: Synch mechanism used to ensure that worker-threads are alive *before*
--          main-thread starts sending them signals. Every main-thread function
--          that somehow handles worker-threads should use this mechanism to
--          avoid deadlocks.
-- Note: This is not technically a Barrier in the pthread sense. Usually, a
--       Barrier would not make distinctions between main- and worker-threads
--       and also it would reset after having been reached by all threads.
--       Here, the barrier is blocking only for the main-thread (it actually
--       has to wait for all threads to come to life) but the
--       worker-threads are allowed to continue after reaching the barrier because
--       there is no reason for them to wait.
--       Furthermore, the barrier is not 'reset', i.e. if the main-thread reaches
--       another call of :wait(), then there is no need to wait for all threads
--       to come to life *again* (makes no sense).
-- init by: initThreads (on main thread)
-- destroyed by: joinThreads (on main thread)
-- used by: (together, wait:main, signal:worker)
-- * GPULauncher(): wait until all threads are alive before putting work into
--                   job-queue (wait until numthreadsAlive==numthreads)
-- * waitForWork(): increase numthreadsAlive before entering infinite loop
-- * joinThreads(): same as GPULaucher.
local numthreadsAliveMutex = global(C.pthread_mutex_t, nil, "numthreadsAlive")
local numthreadsAlive = global(int, 0, "numthreadsAlive")
tp.numthreadsAliveMutex = numthreadsAliveMutex
tp.numthreadsAlive = numthreadsAlive
-- TODO this is awful programming style, change it!
-- TODO implement singleton pattern for all synch mechanisms
local struct ThreadsAliveBarrier {
  -- 'members' are the global from above
}
tp.ThreadsAliveBarrier = ThreadsAliveBarrier
local theThreadsAliveBarrier = global(ThreadsAliveBarrier, nil, 'theThreadsAliveBarrier')
tp.theThreadsAliveBarrier = theThreadsAliveBarrier
terra ThreadsAliveBarrier:init()
  numthreadsAlive = 0
  pt( C.pthread_mutex_init(&numthreadsAliveMutex, nil) )
end
terra ThreadsAliveBarrier:destroy()
  pt( C.pthread_mutex_destroy(&numthreadsAliveMutex) )
end
terra ThreadsAliveBarrier:wait()
  while true do -- use spinlock
    C.pthread_mutex_lock(&numthreadsAliveMutex)
    var numalive = numthreadsAlive
    C.pthread_mutex_unlock(&numthreadsAliveMutex)

    if numalive == numthreads then
      break
    end
  end -- end of spinlock check
end
terra ThreadsAliveBarrier:signal()
  C.pthread_mutex_lock(&numthreadsAliveMutex)
  numthreadsAlive = numthreadsAlive + 1
  C.pthread_mutex_unlock(&numthreadsAliveMutex)
end


-- NOTE: Below the definition the TaskQueue_t, there is a global() declaration
-- of the (globally available) taskQueue. We can't put it here because the
-- above globals are necessary for the definition of TaskQueue_t.
--------------------------------- GLOBALS END                  

--------------------------------- Task_t START
local struct Task_t {
  taskfunction : {&opaque} -> {bool}
  pd : &opaque
}
tp.Task_t = Task_t

terra Task_t:run() : bool
  debm( C.printf('Task_t:run(): starting\n') )
  debm( C.printf('Task_t:run(): self.taskfunction points to %d\n',
                 self.taskfunction) )

  var name = I.__itt_string_handle_create('run_task')
  var domain = I.__itt_domain_create("Main.Domain")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
  I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)


  var moreWorkWillCome = self.taskfunction(self.pd)

  debm( C.printf('Task_t:run(): stopping\n') )
  return moreWorkWillCome
end
--------------------------------- Task_t END

--------------------------------- TaskQueue_t START
local TaskQueue_t = terralib.types.newstruct("TaskQueue_t")
tp.TaskQueue_t = TaskQueue_t
TaskQueue_t.entries:insert({ type = Task_t[numthreads], field = "threadTasks"})

terra TaskQueue_t:get(threadIndex : int)
  return self.threadTasks[threadIndex]
end                                                                                                                                                                                                                                            
terra TaskQueue_t:set(threadIndex : int, task : Task_t)
  var domain = I.__itt_domain_create("Main.Domain")
  debm( C.printf('TaskQueue_t:set(): starting\n') )

  -- insert task into "queue"
  self.threadTasks[threadIndex] = task


  debm( C.printf('TaskQueue_t:set(): starting to lock thread_busy_mutex[%d],\
                 its value is %d\n', threadIndex, thread_busy_mutex[0]) )

  -- var name_signal = 
  --    I.__itt_string_handle_create('TaskQueue_t:set(): sending work signal')
  -- I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_signal)

  -- send signal that work is available START
  -- pt( C.pthread_mutex_lock(&thread_busy_mutex[threadIndex]))

  -- debm( C.printf('TaskQueue_t:set(): signaling that work is\
  --                 available for thread %d\n', threadIndex) )

  -- pt( C.pthread_cond_signal(&work_available_cv[threadIndex]))

  -- I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_signal)

  -- debm( C.printf('TaskQueue_t:set(): starting to unlock\
  --                 thread_busy_mutex[%d]\n', threadIndex) )
  -- pt( C.pthread_mutex_unlock(&thread_busy_mutex[threadIndex]))
  theWaitForWorkSemaphore:signal(threadIndex)
  -- send signal that work is available END

  debm( C.printf('TaskQueue_t:set(): stopping\n') )
end 

taskQueue = global(TaskQueue_t, nil, "taskQueue")
--------------------------------- TaskQueue_t END

-- threadpool stuff start
local terra waitForWork(arg : &opaque) : &opaque                                      
  var threadIndex = [int64](arg)                                                
  debm( C.printf("waitForkWork(tid=%d): starting\n", threadIndex) )
  debm( C.printf("waitForkWork(tid=%d): locking thread_busy_mutex[%d],\
                  value before locking is %d\n",
                  threadIndex, threadIndex, thread_busy_mutex[threadIndex]) )
  debm( C.printf("waitForkWork(tid=%d): locking thread_busy_mutex[%d],\
                  value after locking is %d\n",
                  threadIndex, threadIndex, thread_busy_mutex[threadIndex]) )
  -- pt( C.pthread_mutex_lock(&thread_busy_mutex[threadIndex]) )
  theWaitForWorkSemaphore:initialLock(threadIndex)
  -- while numwloads_finished < NUMWLOADS  do                                   
    -- C.sleep(1)

  var moreWorkWillCome = true

  theThreadsAliveBarrier:signal()

  while moreWorkWillCome  do                                                                
    debm( C.printf("waitForkWork(tid=%d): starting to wait for work,\
                    the value of thread_busy_mutex[%d] is %d\n",
                    threadIndex, threadIndex, thread_busy_mutex[threadIndex]) )

    -- debm( C.printf("waitForkWork(tid=%d): sending readyforwork signal\n", threadIndex, threadIndex, thread_busy_mutex[threadIndex]) )

    var name = I.__itt_string_handle_create('wait_for_work')
    var name2 = I.__itt_string_handle_create('pthread_exit')
    var domain = I.__itt_domain_create("Main.Domain")
    I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)

    -- unlocks mutex on entering, locks it on exit)
    -- pt( C.pthread_cond_wait(&work_available_cv[threadIndex],
    --     &thread_busy_mutex[threadIndex]))
    theWaitForWorkSemaphore:wait(threadIndex)


    I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)


    debm( C.printf("waitForkWork(tid=%d): after receiving signal for work,\
                    value of thread_busy_mutex is %d\n",
                    threadIndex, thread_busy_mutex[threadIndex]) )

    debm( C.printf("waitForkWork(tid=%d): receiving work\n", threadIndex) )
    var task = taskQueue:get(threadIndex)                                       

    debm( C.printf("waitForkWork(tid=%d): running work\n", threadIndex) )
    moreWorkWillCome = task:run()                                                        
    debm( C.printf("waitForkWork(tid=%d): finished running work\n", threadIndex))
  end                                                                           


  debm( C.printf("waitForkWork(tid=%d): unlocking thread_busy_mutex[%d]\n",
                  threadIndex, threadIndex))
  -- pt( C.pthread_mutex_unlock(&thread_busy_mutex[threadIndex]) )
  theWaitForWorkSemaphore:finalUnlock(threadIndex)

  debm( C.printf("waitForkWork(tid=%d): calling pthread_exit()\n",
                 threadIndex, threadIndex))
  C.pthread_exit(nil)
  return nil                                                                    
end             


-- TODO create corresponding join function and use in init, cost, and step()
local terra initThreads()
  debm( C.printf('initThreads(): starting\n') )
  -- pt( C.pthread_key_create(&tid_key, nil))

  -- for k = 0,numthreads do                                                       
  --   debm( C.printf('initThreads(): value of thread_busy_mutex[%d]\
  --                   before init is %d\n', k, thread_busy_mutex[k]) )
  --   pt( C.pthread_mutex_init(&thread_busy_mutex[k], nil))
  --   debm( C.printf('initThreads(): value of thread_busy_mutex[%d]\
  --                   after init is %d\n', k, thread_busy_mutex[k]) )
  --   pt( C.pthread_cond_init(&work_available_cv[k], nil) )
  -- end                                                                           
  theWaitForWorkSemaphore:init()


  -- pt( C.pthread_mutex_init(&thread_has_been_canceled_mutex, nil) )

  -- pt( C.pthread_mutex_init(&numkernels_finished_mutex, nil) )
  -- pt( C.pthread_mutex_init(&kernel_running_mutex, nil) )
  -- pt( C.pthread_cond_init(&kernel_finished_cv, nil) )
  theKernelFinishedByAllThreadsBarrier:init()

  -- pt( C.pthread_mutex_init(&ready_for_work_mutex, nil) )

  theThreadsAliveBarrier:init()

  -- pt( C.pthread_cond_init(&thread_has_been_canceled_cv, nil) )
  -- pt( C.pthread_cond_init(&ready_for_work_cv, nil) )

  -- set cpu affinities (TODO this is broken, needs to be fixed)
  -- escape
  --   if c.cpumap then
  --     emit quote
  --       var cpusets : C.cpu_set_t[numthreads]
  --       var cpumap : int[8]

  --       escape
  --         for k = 1,numthreads do
  --           emit quote
  --             cpumap[ [k-1] ] = [ c.cpumap[k] ]
  --           end
  --         end
  --       end

  --       -- CPU_ZERO macro -- TODO refactor these macros
  --       for k = 0,numthreads do
  --         C.memset ( &(cpusets[k]) , 0, sizeof (C.cpu_set_t)) -- 0 is the integer value of '\0'
  --       end

  --       -- CPU_SET macro
  --       for k = 0,numthreads do
  --         var cpuid : C.size_t = cpumap[k]
  --         ([&C.__cpu_mask](cpusets[k].__bits))[0] = ([&C.__cpu_mask](cpusets[k].__bits))[0] or ([C.__cpu_mask]( 1  << cpuid) )
  --       end


  --       for k = 0,numthreads do
  --         tdatas[k].cpuset = cpusets[k]
  --       end
  --     end
  --   end
  -- end

  for tid = 0,numthreads do
    pt( C.pthread_create(&theThreads[tid], nil, waitForWork, [&opaque](tid)))
  end
  -- C.sleep(1)
  debm( C.printf('initThreads(): stopping\n') )
end
tp.initThreads = initThreads


-- Definition of a task to get threads out of their infinite while-loop
local terra stopWaitingForWork(dummy : &opaque)
   var moreWorkWillCome = false
   return moreWorkWillCome
end
local stopWaitingForWorkTask = 
   global(Task_t, `Task_t( {taskfunction=stopWaitingForWork, pd = nil} ),
          'stopWaitingForWorkTask')


local terra joinThreads()
  debm( C.printf('joinThreads(): starting\n') )

  -- wait for all threads to start (why are we doing this in jointhreads?)
  var waitForThreadsAliveName = I.__itt_string_handle_create('wait_for_threads_to_start')
  var domain = I.__itt_domain_create("Main.Domain")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, waitForThreadsAliveName)

  theThreadsAliveBarrier:wait()

  I.__itt_task_end(domain, I.__itt_null, I.__itt_null, waitForThreadsAliveName)

  -- send "kill signal" to all threads to get them to exit from their
  -- infinite "waitForWork" state.
  for tid = 0,numthreads do
    taskQueue:set(tid, stopWaitingForWorkTask)
  end

  -- We need to do this here because pthread_join() should be called from
  -- main thread. The worker-threads can't kill themselves.
  debm( C.printf('joinThreads(): waiting for threads to join\n') )
  for tid = 0,numthreads do                                                       
    pt( C.pthread_join(theThreads[tid], nil))
  end
  debm( C.printf('joinThreads(): threads are joined\n') )

  -- cleanup (but necessary because if we forget to destroy then the mutexes
  -- are in an undefined state after restarting the the threads)
  -- for k = 0,numthreads do                                                       
  --   pt( C.pthread_mutex_destroy(&thread_busy_mutex[k]) )
  --   pt( C.pthread_cond_destroy(&work_available_cv[k]))
  -- end                                                                           
  theWaitForWorkSemaphore:destroy()

  debm( C.printf('joinThreads(): bla1\n') )
  -- pt( C.pthread_mutex_destroy(&numkernels_finished_mutex))
  -- pt( C.pthread_mutex_destroy(&kernel_running_mutex) )
  -- pt( C.pthread_cond_destroy(&kernel_finished_cv) )
  theKernelFinishedByAllThreadsBarrier:destroy()
  debm( C.printf('joinThreads(): bla2\n') )

  -- pt( C.pthread_cond_destroy(&thread_has_been_canceled_cv) )
  -- pt( C.pthread_mutex_destroy(&thread_has_been_canceled_mutex) )
  debm( C.printf('joinThreads(): bla3\n') )

  theThreadsAliveBarrier:destroy()
  debm( C.printf('joinThreads(): stopping\n') )
end
tp.joinThreads = joinThreads
-- threadpool stuff end

return tp
