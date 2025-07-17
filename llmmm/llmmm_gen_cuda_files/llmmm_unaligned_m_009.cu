#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_9_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<9> mm_instantiator;

public:
  UnalignedM_9_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_9_MMInstantiatorWrapper__;

}  // namespace LLMMM
