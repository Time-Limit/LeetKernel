#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_87_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<87> mm_instantiator;

public:
  UnalignedM_87_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_87_MMInstantiatorWrapper__;

}  // namespace LLMMM
