#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// merges the component labels along the image columns, where label -1 represents
// the absence of a label
kernel void columnMerge(constant int &width [[buffer(0)]],
                        constant int &height [[buffer(1)]],
                        device int *buffer [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= width)
    {
        return;
    }
    
    int curX = gid.x;
    int curLoc = curX;
    int prev = buffer[curLoc];
    for (int i=1; i<height; i++)
    {
        curLoc += width;
        int cur = buffer[curLoc];
        if (prev != -1 && cur != -1) {
            cur = prev;
            buffer[curLoc] = cur;
        } else
        {
            prev = cur;
        }
    }
}

//each getRoot and getRootVolatile return the root label for a buffer location

int getRootVolatile(volatile device atomic_int* buffer, int loc)
{
    int label = atomic_load_explicit(buffer+loc, memory_order::memory_order_relaxed);
    if (label == -1)
    {
        return -1;
    }
    int prevLabel = loc;
    
    while (label != prevLabel)
    {
        prevLabel = label;
        label = atomic_load_explicit(buffer+label, memory_order::memory_order_relaxed);
    }
    return label;
}

int getRoot(device int* buffer, int loc)
{
    int label = buffer[loc];
    if (label == -1)
    {
        return -1;
    }
    int prevLabel = loc;
    
    while (label != prevLabel)
    {
        prevLabel = label;
        label = buffer[label];
    }
    return label;
}

// Merges the component labels along the image rows, producing blockSize . To finish the merging, call
// again using only one threadGroup that spans the whole width of the image, and
// with mergeSize equal to the size of the blocks of columns previously merged.
kernel void connectedCompReduction(constant int& width [[buffer(0)]],
                          constant int& height [[buffer(1)]],
                          constant int& mergeSize [[buffer(2)]],
                          volatile device atomic_int* buffer [[buffer(3)]],
                          constant int& blockSize [[buffer(4)]],
                          uint bid [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]])
{
    //int pixXBase = mergeSize-1 + 2*mergeSize*gidX;
    int threadGroupPixWidth = 2*mergeSize*blockSize;
    int threadGroupBase = threadGroupPixWidth*bid-1; // base for 1-indexed access of block in buffer
    int overallLim = min(width, threadGroupBase + threadGroupPixWidth);
    int pixXOffset = mergeSize + 2*mergeSize*tid; // 1-indexed offset for access of block in buffer
    for (int mergeSizeMult = 1; mergeSizeMult < 2*blockSize; mergeSizeMult *= 2) {
        int pixX = threadGroupBase + pixXOffset*mergeSizeMult;
        if (pixX < overallLim)
        {
            for (int gidY = 0; gidY < height; gidY++)
            {
                int loc = gidY*width + pixX;
                int rLoc = loc + 1;
                
                int lVal = atomic_load_explicit(buffer+loc, memory_order::memory_order_relaxed);
                int rVal = atomic_load_explicit(buffer+rLoc, memory_order::memory_order_relaxed);
                
                if (lVal != -1 && rVal != -1)
                {
                    lVal = getRootVolatile(buffer, lVal);
                    rVal = getRootVolatile(buffer, rVal);
                    if (lVal > rVal)
                    {
                        atomic_store_explicit(buffer+lVal, rVal, memory_order::memory_order_relaxed);
                    } else
                    {
                        atomic_store_explicit(buffer+rVal, lVal, memory_order::memory_order_relaxed);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }
}

// records the root of each pixel, traversing the union-find tree,
// and creates a sparse set of the component labels present in the components
// buffer
kernel void assignComp(constant int& width [[buffer(0)]],
                       constant int& height [[buffer(1)]],
                       device int* buffer [[buffer(2)]],
                       device int* components [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]])
{
    int gidX = gid.x;
    int gidY = gid.y;
    
    if (gidX >= width || gidY >= height)
    {
        return;
    }
    
    int loc = gidY * width + gidX;
    int root = getRoot(buffer, loc);
    buffer[loc] = root;
    if (root != -1)
    {
        components[root] = root+1;
    }
}

// consolidates the inBuffer sparse set into a dense set of all component labels
// present in the connected component labeling
kernel void condense(device int* inBuffer [[buffer(0)]],
                     constant uint& bufferWidth [[buffer(1)]],
                     constant uint& bufferHeight [[buffer(2)]],
                     constant uint& startMergeSize [[buffer(3)]],
                     constant uint& blockDim [[buffer(4)]],
                     uint bid [[ threadgroup_position_in_grid ]],
                     uint tid [[ thread_index_in_threadgroup ]])
{
    uint bufferSize = bufferWidth * bufferHeight;
    // position of threadgroup in whole buffer
    uint startInd = bid*blockDim*2;
    // limit on in-range elements of blockDim
    uint lim1 = min(blockDim*2, bufferSize - startInd);
    // mergeSize: size of halves to be merged
    for (uint mergeSize=startMergeSize; mergeSize<lim1; mergeSize *= 2)
    {
        // position of merge segment start in buffer group
        uint pos = tid*mergeSize*2;
        // position of merge segment start in whole buffer
        uint bufPos = startInd + pos;
        // must be at position less than size of block
        if (pos < blockDim*2 && bufPos < bufferSize)
        {
            // current buffer position
            uint curPos = bufPos;
            // position of beginning of right half of merge segment
            uint rightPos = bufPos + mergeSize;
            uint rightLim = min(rightPos+mergeSize, rightPos + (lim1 - (pos + mergeSize)));
            while (curPos < rightPos && inBuffer[curPos] != 0)
            {
                curPos++;
            }
            if (curPos < rightPos)
            {
                while (rightPos < rightLim && inBuffer[rightPos] != 0) {
                    inBuffer[curPos] = inBuffer[rightPos];
                    inBuffer[rightPos] = 0;
                    curPos++;
                    rightPos++;
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_none);
    }
}
