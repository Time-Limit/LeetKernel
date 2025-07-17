#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_120_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<120> mm_instantiator;

public:
  UnalignedM_120_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_120_MMInstantiatorWrapper__;

}  // namespace LLMMM
