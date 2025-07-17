#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_7_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<7> mm_instantiator;

public:
  UnalignedM_7_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_7_MMInstantiatorWrapper__;

}  // namespace LLMMM
