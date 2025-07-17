#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_115_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<115> mm_instantiator;

public:
  UnalignedM_115_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_115_MMInstantiatorWrapper__;

}  // namespace LLMMM
